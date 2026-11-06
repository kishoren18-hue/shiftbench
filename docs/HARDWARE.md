# What to buy

Everything for ShiftBench, plus what SteadyHand 2 adds later. The robotic hand
is not on this list (you're choosing that yourself). The SO-101 gripper can
stand in for it in a demo (`steadyhand2/hand.py`).

Prices marked **verified** come from a source checked on Nov 6, 2026. Prices
marked *est.* are rough; check before buying.

## ShiftBench (buy now)

| # | Item | Why this one | Price |
|---|---|---|---|
| 1 | **SO-ARM101 Pro kit, leader + follower** (Seeed Studio or its AliExpress store) | The **Pro** follower has 12 V motors with 30 kg·cm torque vs 16.5 kg·cm on the Standard. It's $20 more, and the extra margin helps when the gripper doubles as SteadyHand's hand. The leader arm is how you teleoperate the 50 demos. | **$240** verified (Standard $220) |
| 2 | **3D-printed parts** for both arms | Not included in either kit. Add Seeed's printed-parts option, or print them free at a school or library makerspace (PLA, STLs in TheRobotStudio's SO-ARM100 GitHub repo). | **~$35** option, verified |
| 3 | **Webcam with manual exposure, x2** (e.g. Logitech C920-class) | One fixed **front** camera (the one ShiftBench shifts) and one **wrist** or top camera. It must let you **lock exposure and white balance** (`v4l2-ctl` on Linux, or the vendor app). Auto-exposure partly undoes a lighting change, which muddies the experiment. | *est. $30–70 each* |
| 4 | **Wrist camera mount** | The SO-ARM100 repo has a wrist-camera mount STL. Check which camera module it fits before buying #3's second camera. | print it |
| 5 | **Small tripod + ball head** for the front camera | Makes camera shifts repeatable: tape marks on the table for position, a protractor sticker on the head for angle. | *est. $15–25* |
| 6 | **App-controlled smart bulb + clamp lamp** | Lighting levels become numbers ("40%, 2700 K") anyone can reproduce. | *est. $15–30* |
| 7 | **Felt sheets** (grey and blue) + **wood-grain contact paper** | Table shifts. Grey felt is nominal. | *est. $10–15* |
| 8 | **Task objects**: 40 mm foam cubes (red + blue), a small bowl, plus a toy and a glue stick | Red cube into bowl is the nominal task. The rest are new-object shifts. | *est. $10–15* |
| 9 | **2 C-clamps** (skip if your kit includes table clamps) | The arm base must not move between sessions, or every session is an accidental shift. | *est. $10* |
| 10 | **Powered USB hub** + a spare USB-C cable | Two arms + two cameras = four USB devices. Unpowered hubs drop cameras. | *est. $20–30* |
| — | Lux meter app on your phone | Record lux at the cube for every session. | free |
| — | Google Colab | Training and encoding. Free T4 works; Colab Pro only if sessions keep timing out. | free / ~$10/mo |

**Rough total for ShiftBench: about $420–560**, depending on cameras and what you print yourself.

## SteadyHand 2 (later, besides the hand)

| Item | Note |
|---|---|
| **EMG armband with a LibEMG streamer** | LibEMG 2.0.3 (the version SteadyHand pins) has streamers for SiFi BioArmband/BioPoint, OyMotion gForce, MindRove, Myo, Delsys, EMaGer and OTB Muovi+. The Myo is discontinued. OyMotion gForce Pro+ (8 ch, up to 1 kHz) and MindRove (8 ch, Wi-Fi) are the usual affordable picks. Check current prices yourself; I couldn't verify them. On Linux, the OyMotion streamer needs its USB dongle and sudo. |
| **Wrist camera** | The second webcam from #3 works if it mounts near your wrist. A small USB camera module on a strap is better. |

## Before you order

* Buy from Seeed directly or its official AliExpress store. Resellers listed the same Pro kit at $312 and up.
* If you only have budget for one robot purchase right now, it's #1 + #2. Everything else is cheap and can wait a week.
