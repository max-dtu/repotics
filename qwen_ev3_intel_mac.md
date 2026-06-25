# Qwen EV3 on This Intel Mac

This machine is an Intel MacBook Pro with:

- 32 GB RAM
- 2.4 GHz 8-Core Intel Core i9
- Intel UHD Graphics 630
- AMD Radeon Pro 5500M with 8 GB VRAM
- Metal 3 support

## Recommended First Run

Use CPU first. It is slower, but it is the most reliable path on this Intel Mac.

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1 --device cpu
```

## Optional MPS / AMD GPU Run

If PyTorch exposes the AMD GPU through MPS, you can try Metal acceleration:

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1 --device mps --torch-dtype float16
```

MPS does not support `bfloat16`, so force `float16` when trying this path.
If it fails with unsupported operations, memory errors, or dtype errors, go back
to the CPU command above.

## Useful Test Commands

Dry run one model decision without sending anything to the EV3:

```bash
python qwen_ev3_loop.py --dry-run --max-steps 1 --device cpu
```

Use the local Hugging Face cache after the model has already downloaded:

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1 --device cpu --local-files-only
```

## EV3 Network Troubleshooting

The EV3 server may print more than one IP address. Use the one on the same
subnet as the Mac.

For this Mac, Wi-Fi is on `10.45.151.x`, so use:

```bash
python qwen_ev3_loop.py --host 10.45.151.18 --port 9999 --hz 1 --device cpu
```

Do not use `192.168.0.1` from the Mac unless the Mac also has a `192.168.0.x`
network interface.

Check basic reachability from the Mac:

```bash
ping 10.45.151.18
```

Check that the EV3 TCP server is accepting connections:

```bash
nc -vz 10.45.151.18 9999
```

Send one manual command:

```bash
printf 'forward\n' | nc 10.45.151.18 9999
```

If you see `No route to host`, restart Wi-Fi on the EV3, restart `ev3-side.py`,
and confirm the EV3 still prints `10.45.151.18`. If the IP changed, pass the new
IP to `--host`.

If both devices are on the same phone hotspot and `ping` says `No route to host`,
the hotspot is probably blocking client-to-client traffic. That means both
devices have internet, but the hotspot is not acting like a normal local LAN.

To keep using local TCP, use one of these local network setups:

- Put both devices on a normal Wi-Fi router that allows device-to-device traffic.
- Use EV3 USB networking and connect to `192.168.0.1`.
- Use a phone hotspot only if it has a setting that allows local device access
  or disables client isolation.

## Local USB Networking

The EV3 often exposes a USB network interface at `192.168.0.1`. Connect the EV3
to the Mac over USB, then give the Mac side of that USB interface an address on
the same subnet.

First find the USB interface on the Mac:

```bash
ifconfig
```

In the current setup, the likely USB interface is `en5`. Configure it manually:

```bash
sudo ifconfig en5 inet 192.168.0.2 netmask 255.255.255.0 up
```

Then test local reachability:

```bash
ping 192.168.0.1
nc -vz 192.168.0.1 9999
```

Run Qwen locally over USB:

```bash
python qwen_ev3_loop.py --host 192.168.0.1 --port 9999 --hz 1 --device cpu
```

## Local SSH Tunnel

If SSH works but direct TCP to `10.45.151.18:9999` does not, keep the EV3 server
running in one EV3 terminal:

```bash
python3 ev3-side.py
```

Open a second Mac terminal and create a local tunnel:

```bash
ssh -N -L 10099:127.0.0.1:9999 robot@ev3dev.local
```

Leave that terminal open. In another Mac terminal, send a manual command through
the tunnel:

```bash
printf 'forward\n' | nc 127.0.0.1 10099
```

Then run Qwen through the same local tunnel:

```bash
python qwen_ev3_loop.py --host 127.0.0.1 --port 10099 --hz 1 --device cpu
```

This is still local control. It just uses SSH as the transport because
`ev3dev.local` is reachable even when direct IPv4 TCP is not.

## Task Prompting And ReAct Loop

The loop is:

```text
Camera frame
  -> vision-language model observes the scene
  -> model returns command + task_done + short reason
  -> script sends only the command to EV3
  -> repeat until task_done=true
  -> ask for the next task
```

At startup, the script asks for the current robot task:

```text
Robot task [move toward the ball]>
```

Press Enter to use the default, or type a task such as:

```text
move toward the ball
push the ball into the goal
turn until you can see the ball
```

Run without interactive task prompting:

```bash
python qwen_ev3_loop.py --host 127.0.0.1 --port 10099 --hz 1 --device cpu --no-ask-task --task "move toward the ball"
```

The prompt asks Qwen to choose exactly one command: `left`, `right`, `forward`,
or `backward`. If unsure, it should rotate left or right.

## Vision Decision Logs

The Qwen loop logs the full decision each cycle:

```text
Vision decision: command=forward car_visible=True ball_visible=True target=ball reason=...
Model raw output: {"command":"forward","car_visible":true,...}
Model frame: qwen_latest_frame.jpg (320x240)
```

Open `qwen_latest_frame.jpg` to inspect exactly what Qwen saw on the latest
iteration. The file is overwritten each cycle.

Disable frame saving:

```bash
python qwen_ev3_loop.py --host 127.0.0.1 --port 10099 --hz 1 --device cpu --save-latest-frame ''
```

## Internet Remote Fallback

Use this only when local networking is blocked or unavailable.

On the EV3, edit `ev3-side.py`:

```python
REMOTE_MODE = True
ROBOT_UNIQUE_ID = "kals-ev3"
```

Start the EV3 listener:

```bash
python3 ev3-side.py
```

On the Mac, run Qwen through the `ntfy` relay:

```bash
python qwen_ev3_loop.py --host ntfy:kals-ev3 --hz 1 --device cpu
```

Send one manual relay command from the Mac:

```bash
python controller.py
```

## Practical Settings

- Keep `--hz` around `1` or `2`.
- Keep image size near the default `320x240`.
- Use the strict one-word command prompt.
- Valid robot commands are `left`, `right`, `forward`, and `backward`.
