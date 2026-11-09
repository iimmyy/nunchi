# Setting up Nunchi for a service

Nunchi runs on the room's own computer. It hears the sermon from the computer's sound input, translates it, and every
phone on the same Wi-Fi opens its page for the English, as text and as a voice in earbuds. Nothing goes to the
internet, and no one installs anything on their phone.

## What you need

- The **Nunchi folder**: `nunchi.exe`, `seoul.gguf`, `seoul-mm.gguf`, `LICENSE-LFM.txt`. About 2 GB, so a USB stick
  is the easy way to bring it.
- The computer with the sermon's sound coming in: the mixer's output into its line or mic input, or a USB audio box.
- A Wi-Fi network that the computer and the phones are both on. It doesn't need internet. Any of these works:
  - a small travel router (the computer plugs into it, phones join its Wi-Fi), the most dependable;
  - the computer's own Mobile Hotspot (Windows Settings, Network, Mobile hotspot), if it has Wi-Fi;
  - a phone's personal hotspot, for a few listeners.

## Before the service

1. Copy the Nunchi folder to the computer (the Desktop is fine).
2. Join the computer to the Wi-Fi (or turn its hotspot on).
3. Double-click `nunchi.exe`.
   - Windows may say "Windows protected your PC": choose **More info**, then **Run anyway**.
   - When Windows asks about the firewall, allow it on **private networks**. Without this, phones can't open the page.
4. A black window shows Nunchi loading (about 10 seconds), then the screen page opens in the browser.
5. On the screen page, pick the input the sermon comes in on, then press **Start**. Allow the microphone when the
   browser asks. The level bar moves when someone speaks.
6. Type the Wi-Fi name and password into the screen page once. Its two QR codes are then ready: the first joins the
   Wi-Fi, the second opens Nunchi.

## For the people listening

1. Scan the first code to join the Wi-Fi, then the second to open Nunchi.
2. The English appears as he speaks. Tap **Listen** for the voice in earbuds.
3. Keep the page open on the screen while listening (phones stop the sound when they lock).

## During the service

- Press **Stop** for songs and anything that shouldn't be translated, and **Start** again for the sermon.
- **Projector view** shows big captions only (Esc to go back).
- Untick **voice** for captions only. That's lighter on a slow computer.
- This week's terms (one per line, `사사기 = Judges`) help with names the translation might not know.

## If something is wrong

| What you see | What to do |
|---|---|
| Phones can't open the page | The computer and the phone must be on the same Wi-Fi; allow Nunchi in the Windows firewall (private networks) |
| The level bar doesn't move | Pick another input on the screen page; check the cable from the mixer |
| The English falls far behind | Untick **voice**; close other programs |
| The black window closed | Double-click `nunchi.exe` again |
