# Setting up Nunchi for a service

Nunchi runs on the room's own computer. It hears the sermon from the computer's sound input, translates it, and every
phone on the same Wi-Fi opens its page for the English, as text and as a voice in earbuds. Nothing goes to the
internet, and no one installs anything on their phone.

## What you need

- The **Nunchi folder**: `nunchi.exe`, `seoul.gguf`, `seoul-mm.gguf`, `LICENSE-LFM.txt`. About 2 GB, so a USB stick
  is the easy way to bring it.
- The computer with the sermon's sound coming in: the mixer's output into its line or mic input, or a USB audio box.
- A Wi-Fi network that the computer and the phones are both on. It doesn't need internet. One of these:
  - a small travel router (the computer joins it by cable or Wi-Fi, phones join its Wi-Fi). The most dependable: it
    needs no internet and takes dozens of phones;
  - a phone's personal hotspot, for a few listeners (about five on an iPhone);
  - the computer's own Mobile Hotspot (Windows Settings, Network, Mobile hotspot). Windows only turns it on while
    the computer itself is on the internet, and it takes 8 phones at most.

## Before the service

1. Copy the Nunchi folder to the computer (the Desktop is fine). Plug the computer in, and if it's a laptop, keep it
   open: Nunchi keeps it awake while its window is open, but a closed lid still puts it to sleep.
2. Join the computer to the Wi-Fi (or turn its hotspot on).
3. Double-click `nunchi.exe`.
   - Windows may say "Windows protected your PC": choose **More info**, then **Run anyway**.
   - When Windows asks about the firewall, tick both boxes (**private** and **public networks**) and choose
     **Allow**. Windows calls a new Wi-Fi public, and without this, phones can't open the page.
4. A black window shows Nunchi loading (about 10 seconds), then the screen page opens in the browser.
5. On the screen page, pick the input the sermon comes in on, then press **Start**. Allow the microphone when the
   browser asks. The level bar moves when someone speaks.
6. Type the Wi-Fi name and password into the screen page once. Its two QR codes are then ready: the first joins the
   Wi-Fi, the second opens Nunchi.

## For the people listening

1. Scan the first code to join the Wi-Fi, then the second to open Nunchi.
2. Tap **Listen and read** for the voice in earbuds with the English on screen, or **Read only**. The English appears
   as he speaks, and the phone's screen stays on while the page is open.
3. Keep Nunchi open in front: switching apps or locking the phone stops the sound until it's opened again.

## During the service

- Press **Stop** for songs and anything that shouldn't be translated, and **Start** again for the sermon.
- **Projector view** shows big captions only (Esc to go back).
- Untick **voice** for captions only. That's lighter on a slow computer.
- This week's terms (one per line, `사사기 = Judges`) help with names the translation might not know.

## If something is wrong

| What you see | What to do |
|---|---|
| Phones can't open the page | The computer and the phone must be on the same Wi-Fi. Then the firewall: Windows Security, Firewall, Allow an app through firewall, and tick both boxes next to nunchi |
| An Android phone says the Wi-Fi has no internet | Tap **Stay connected** (or turn mobile data off for the service); otherwise it sends the page to mobile data, where Nunchi isn't |
| The level bar doesn't move | Pick another input on the screen page; check the cable from the mixer |
| The English falls far behind | Untick **voice**; close other programs |
| The black window closed | Double-click `nunchi.exe` again |
