#include <OptiPrime/Protocols.h>
#include <Model/FaceResNet.h>
#include <LinearLayer/ConvOptiEncode.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using RingTensor = Datatype::Tensor<uint64_t>;
using Face = Model::FaceResNet64;
using Clock = std::chrono::steady_clock;
using namespace OptiPrime;
constexpr int bits = 20;
constexpr uint64_t mask = (uint64_t(1) << bits) - 1;

struct Compact {
    Context &ctx;
    bool light;
    std::vector<Face::ConvShape> shapes = Face::ConvolutionShapes(Face::Backbone::Compact);
    std::vector<std::unique_ptr<LinearLayer::Conv2DOptiEncode>> layers;

    Compact(Context &c, bool approximate, const std::filesystem::path &root) : ctx(c), light(approximate) {
        LinearLayer::OptiEncodeOptions opt;
        opt.threads = 16;
        opt.ciphertext_inverse_normalization = true;
        opt.compact_output = true;
        opt.add_bias = false;
        std::ifstream file;
        if (ctx.config.party == 1) {
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.open(root/"weights.bin",std::ios::binary);
        }
        for (const auto &s: shapes) {
            RingTensor w({s.out_channels,s.in_channels,s.kernel,s.kernel}), b({s.out_channels});
            if (ctx.config.party == 1) {
                std::vector<int8_t> signed_weights(w.size());
                file.read(reinterpret_cast<char*>(signed_weights.data()),signed_weights.size());
                for (size_t j=0;j<w.size();++j) w(j)=uint64_t(signed_weights[j]) & mask;
            }
            layers.emplace_back(std::make_unique<LinearLayer::Conv2DOptiEncode>(
                s.size,s.stride,s.kernel/2,w,b,ctx.he.get(),opt));
        }
    }

    RingTensor conv(size_t i, RingTensor x, bool relu, int low, int high) {
        const auto &s=shapes[i];
        ctx.RingToField(x);
        auto y=(*layers[i])(x);
        ctx.FieldToRing(y);
        size_t plane=y.size()/s.out_channels;
        // The first two channels carry the large central coefficients in the
        // benchmark fixture. The remaining channels have scale 2^-14.
        for (size_t g=0;g<2;++g) {
            size_t begin=g==0?0:2*plane, end=g==0?2*plane:y.size();
            int shift=g==1?14:(i==0 || s.kernel==1?2:4);
            RingTensor part({end-begin});
            std::copy(y.data().begin()+begin,y.data().begin()+end,part.data().begin());
            if (light && relu) {
                part=Requantize(ctx,std::move(part),shift,0);
            } else {
                ctx.fix->truncate(part,shift,bits,false,true);
                part=Clamp(ctx,part,low,high,bits);
            }
            std::copy(part.data().begin(),part.data().end(),y.data().begin()+begin);
        }
        return y;
    }

    RingTensor forward(const RingTensor &input) {
        auto x=conv(0,input,true,0,255);
        size_t layer=1;
        for (int stage=0;stage<3;++stage) {
            auto y=conv(layer++,x,true,0,255);
            y=conv(layer++,std::move(y),false,-128,127);
            auto skip=stage?conv(layer++,x,false,-128,127):std::move(x);
            x=y+skip;
            x=light?LowerRelu(ctx,x):Clamp(ctx,x,0,255,bits);
        }
        RingTensor sums({64,1,1});
        if (x.shape()!=std::vector<size_t>{64,16,16}) throw std::runtime_error("pool shape");
        for (size_t c=0;c<64;++c)
            for (size_t j=0;j<256;++j) sums(c)+=x(c*256+j);
        ctx.fix->truncate(sums,8,bits,true,false);
        auto out=conv(layer,sums,false,-64,64);
        out.reshape({128});
        ctx.ExtendSigned(out,bits,Face::bitwidth);
        for (auto &v:out.data()) v*=64; // Q6 -> Q12 for the original cosine matcher
        return out;
    }
};

RingTensor input_share(int party,int sign,int ordinal) {
    RingTensor x({3,64,64});
    for (size_t i=0;i<x.size();++i) {
        uint64_t fixture_share=(i*0x9e3779b97f4a7c15ULL+ordinal*0xbf58476d1ce4e5b9ULL) & mask;
        uint64_t code=i<4096?sign*64:0;
        x(i)=party==1?fixture_share:(code-fixture_share)&mask;
    }
    return x;
}
std::vector<int64_t> check_embedding(Context &ctx,const RingTensor &x,int sign,bool light,const std::filesystem::path &root) {
    constexpr uint64_t modulus=uint64_t(1)<<Face::bitwidth;
    auto *io=ctx.he->IO;
    if (ctx.config.party==2) {
        io->send_data(x.data().data(),128*sizeof(uint64_t));io->flush();
        return {};
    }
    std::vector<uint64_t> other(128);
    io->recv_data(other.data(),128*sizeof(uint64_t));
    std::ifstream f(root/("ref-"+std::to_string(sign)+"-"+std::to_string(int(light))+".bin"),std::ios::binary);
    std::vector<int64_t> bounds(256),values(128);
    f.read(reinterpret_cast<char*>(bounds.data()),256*sizeof(int64_t));
    if(!f) throw std::runtime_error("reading plaintext reference");
    for(size_t i=0;i<128;++i) {
        auto u=(other[i]+x(i))&(modulus-1);
        auto v=u>=modulus/2?int64_t(u)-int64_t(modulus):int64_t(u);
        if(v<bounds[i] || v>bounds[128+i]) throw std::runtime_error("embedding disagrees at "+std::to_string(i)+": "+std::to_string(v)+" vs "+std::to_string(bounds[i])+".."+std::to_string(bounds[i+128]));
        values[i]=v;
    }
    return values;
}
double sec(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double>(b-a).count();}
int main(int argc,char **argv) {
    try {
        if(argc!=5) throw std::runtime_error("role port light fixture");
        Config c;c.party=std::stoi(argv[1]);c.port=std::stoi(argv[2]);c.threads=3;
        c.ot=OTBackend::Ferret;c.linear=LinearBackend::CPU;c.preprocess=false;
        c.approximate_relu=std::stoi(argv[3]);c.power_of_two=c.ciphertext_inverse=c.compact_output=true;
        c.activation_ring_bits=c.accumulator_bits=c.plaintext_bits=bits;c.ciphertext_prime_bits={50,50};
        auto setup=Clock::now();
        Context ctx(c);
        ctx.Preprocess(); // live Ferret; no precomputed per-image OT buffers
        auto packed=Clock::now();
        Compact model(ctx,c.approximate_relu,argv[4]);
        std::cout<<"SETUP "<<sec(setup,packed)<<" PACK "<<sec(packed,Clock::now())<<std::endl;
        for (int pair=0;pair<2;++pair) {
            int sy=pair?1:-1;
            auto x=input_share(c.party,1,2*pair), y=input_share(c.party,sy,2*pair+1);
            ctx.Sync();
            auto b0=ctx.Bytes();auto t0=Clock::now();
            auto a=model.forward(x);auto t1=Clock::now();
            auto b=model.forward(y);auto t2=Clock::now();
            uint8_t share=Face::MatchEmbeddings(a,b,0.5,*ctx.fix),peer=0;
            auto *io=ctx.he->IO;
            if(c.party==1){io->send_data(&share,1);io->flush();io->recv_data(&peer,1);}
            else{io->recv_data(&peer,1);io->send_data(&share,1);io->flush();}
            auto end=Clock::now();auto sent=ctx.Bytes()-b0;
            if(int(share^peer)!=pair) throw std::runtime_error("incorrect verification");
            std::cout<<std::setprecision(9)<<"RESULT "<<pair<<" "<<sec(t0,t1)<<" "<<sec(t1,t2)<<" "<<sec(t0,end)<<" "<<sent<<" "<<int(share^peer)<<std::endl;
            auto va=check_embedding(ctx,a,1,c.approximate_relu,argv[4]);
            auto vb=check_embedding(ctx,b,sy,c.approximate_relu,argv[4]);
            if(c.party==1){
                int64_t dot=0,aa=0,bb=0;
                for(size_t j=0;j<va.size();++j){dot+=va[j]*vb[j];aa+=va[j]*va[j];bb+=vb[j]*vb[j];}
                if((dot>0 && 4*(__int128)dot*dot>=(__int128)aa*bb)!=bool(pair)) throw std::runtime_error("clear cosine mismatch");
            }
        }
        std::cout<<"PASS"<<std::endl;
    }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
