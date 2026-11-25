"""Same-host TCP bandwidth and propagation-delay simulation for both parties."""

import asyncio

CONNECTIONS = 3  # The face benchmark uses three OT connections, starting at p+1.


class Direction:
    """Bandwidth scheduler shared by all connections in one direction."""

    def __init__(self, mbps, rtt_ms):
        self.bytes_per_second = mbps * 1_000_000 / 8
        self.propagation_seconds = rtt_ms / 2000
        self.available_at = 0.0

    def delivery_time(self, now, size):
        self.available_at = max(now, self.available_at) + size / self.bytes_per_second
        return self.available_at + self.propagation_seconds


async def relay(reader, writer, direction):
    queue = asyncio.Queue(maxsize=32)
    clock = asyncio.get_running_loop().time

    async def receive():
        while data := await reader.read(64 * 1024):
            await queue.put((direction.delivery_time(clock(), len(data)), data))
        await queue.put(None)

    async def send():
        while entry := await queue.get():
            delivery, data = entry
            if delivery > clock():
                await asyncio.sleep(delivery - clock())
            writer.write(data)
            await writer.drain()
        writer.write_eof()

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(receive())
        tasks.create_task(send())


async def start_wan_proxy(server_port, proxy_port, mbps, rtt_ms, timeout, tasks):
    upload = Direction(mbps, rtt_ms)
    download = Direction(mbps, rtt_ms)

    async def proxy(client_reader, client_writer, port):
        server_writer = None
        try:
            # Each party initializes OT before opening its next connection.
            async with asyncio.timeout(timeout):
                while True:
                    try:
                        server_reader, server_writer = await asyncio.open_connection("127.0.0.1", port)
                        break
                    except ConnectionRefusedError:
                        await asyncio.sleep(0.02)
            async with asyncio.TaskGroup() as streams:
                streams.create_task(relay(client_reader, server_writer, upload))
                streams.create_task(relay(server_reader, client_writer, download))
        finally:
            client_writer.close()
            if server_writer is not None:
                server_writer.close()

    def handler(port):
        def accept(reader, writer):
            tasks.create_task(proxy(reader, writer, port))
        return accept

    listeners = []
    try:
        for offset in range(1, CONNECTIONS + 1):
            listener = await asyncio.start_server(
                handler(server_port + offset), "127.0.0.1", proxy_port + offset)
            listeners.append(listener)
        return listeners
    except BaseException:
        for listener in listeners:
            listener.close()
        raise
