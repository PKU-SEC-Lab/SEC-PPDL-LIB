"""Check two synthetic face pairs with the two-party Compact W4A8 benchmark."""

import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from wan import start_wan_proxy


async def run(args, directory):
    processes = []
    listeners = []

    async def party(role, port, cpus):
        command = [str(args.binary), str(role), str(port),
                   str(int(args.mode == "light")), str(directory)]
        if cpus:
            command = ["taskset", "-c", cpus, *command]
        path = directory / f"party{role}.log"
        with path.open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        if await process.wait():
            raise RuntimeError(f"Party {role} failed. See {path}")

    # The proxy helper binds its three ports starting at base + 1.
    client_port = args.port + 20 if args.wan else args.port
    try:
        async with asyncio.timeout(600), asyncio.TaskGroup() as proxy_tasks:
            if args.wan:
                listeners = await start_wan_proxy(args.port - 1, client_port - 1,
                                                  400, 20, 600, proxy_tasks)
            try:
                async with asyncio.TaskGroup() as parties:
                    parties.create_task(party(1, args.port, args.server_cpus))
                    parties.create_task(party(2, client_port, args.client_cpus))
            finally:
                for listener in listeners:
                    listener.close()
    finally:
        for process in processes:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
        await asyncio.gather(*(process.wait() for process in processes))
        await asyncio.gather(*(listener.wait_closed() for listener in listeners))


def summarize(directory):
    results = []
    for role in (1, 2):
        lines = (directory / f"party{role}.log").read_text().splitlines()
        rows = [line.split() for line in lines if line.startswith("RESULT ")]
        if "PASS" not in lines or [row[1] for row in rows] != ["0", "1"]:
            raise RuntimeError(f"Incomplete verification by party {role}")
        results.append(rows)
    forward_times = []
    pair_times = []
    sent_bytes = 0
    for server, client in zip(*results):
        for column in (2, 3):
            forward_times.append(max(float(server[column]), float(client[column])))
        pair_times.append(max(float(server[4]), float(client[4])))
        sent_bytes += int(server[5]) + int(client[5])
    return {"forward_s": sum(forward_times) / len(forward_times),
            "pair_s": sum(pair_times) / len(pair_times), "pairs_s": pair_times,
            "communication_MiB_per_pair": sent_bytes / len(pair_times) / 2**20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mode", choices=("exact", "light"), required=True)
    parser.add_argument("--wan", action="store_true", help="400 Mb/s per direction, 20 ms RTT")
    parser.add_argument("--port", type=int, default=35921)
    parser.add_argument("--server-cpus", help="optional taskset CPU list")
    parser.add_argument("--client-cpus", help="optional taskset CPU list")
    parser.add_argument("--output", type=Path, default=Path("build/face-w4a8"))
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    if not args.binary.is_file() or not 1 <= args.port <= 65513:
        parser.error("provide a built face_w4a8 binary and a base port between 1 and 65513")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{args.mode}-", dir=args.output)).resolve()
    print(f"Run directory: {directory}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("reference.py")), str(directory)],
                   check=True)
    asyncio.run(run(args, directory))
    summary = {"mode": args.mode, "wan": args.wan, **summarize(directory)}
    report = json.dumps(summary, indent=2)
    (directory / "summary.json").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
