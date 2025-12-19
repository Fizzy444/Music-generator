# Music-generator
🎶 Music Generator Model

This project is a deep learning–based music generation system that learns patterns from MIDI files and generates new musical sequences. Using a Transformer-based architecture, the model analyzes note sequences, timing, and structure to create original melodies.
📁 What is notes.pkl?

notes.pkl is a serialized (pickled) file that stores all the musical notes and chords extracted from the MIDI dataset during preprocessing.

    Content: Each entry represents a note (e.g., C4) or a chord (stored as dot-separated MIDI numbers, e.g., 60.64.67) from the training files.

    Purpose: It acts as the training "corpus," allowing the model to build a vocabulary of unique musical elements and convert them into numerical sequences without reprocessing raw MIDI files every time.

🔊 Audio Conversion & Synthesis

While the model generates MIDI data (instructions), converting those instructions into a high-quality MP3 requires a synthesis engine. This project uses the following software stack:
1. Software Requirements

To convert MIDI to MP3, you must install these tools on your system and add them to your System PATH:

    FluidSynth: A real-time software synthesizer that converts MIDI into audio (WAV) using a SoundFont.

    FFmpeg: A powerful multimedia framework used here to encode the raw audio (WAV) into a compressed MP3 format.

2. The SoundFont (font.sf2)

MIDI files do not contain actual sound; they only contain "notes." A SoundFont is a library of actual recorded instrument samples.

    Why it's needed: FluidSynth uses this file to decide what the "Piano" or "Violin" actually sounds like.

    How to get it:

        Download a General MIDI SoundFont (e.g., GeneralUser GS v1.471).

        Place the downloaded file in the root directory of this project.

        Important: Rename the file to exactly font.sf2.

🛠️ Installation & Setup (Windows)

To ensure the output script can generate MP3s, follow these steps:
Step 1: Install FFmpeg & FluidSynth

The easiest way to install these on Windows is via Winget (PowerShell):
code Powershell

    
# Open PowerShell as Administrator and run:
winget install ffmpeg
winget install fluidsynth

  

Alternatively, download the binaries from their official sites and manually add the /bin folders to your Environment Variables.
Step 2: Install Python Dependencies
code Bash

    
pip install torch music21 numpy pydub

  

Step 3: Verify the Setup

Open a terminal and type:

    ffmpeg -version

    fluidsynth --version

If both return a version number, your system is ready!
🚀 How it Works (Workflow)

    Generation: The Transformer model predicts a sequence of 300 notes based on a seed MIDI file.

    MIDI Export: The sequence is saved as a .mid file in the music_generated/ folder.

    Synthesis: The script calls FluidSynth to "perform" the MIDI using font.sf2, creating a lossless .wav file.

    Encoding: FFmpeg (via pydub) converts the WAV file into the final .mp3 and deletes the temporary WAV file.

🔮 Future Development

    Improved Architectures: Implementing LSTMs or more complex Transformer blocks for better long-term coherence.

    Multi-Instrument Support: Generating separate tracks for drums, bass, and melody.

    Dynamics & Velocity: Training the model to understand "volume" (velocity) to make the music sound more human.

    Web Interface: A simple GUI or web app to upload a seed and download the generated MP3.

📜 License

This project is open-source. Feel free to use and modify it for your own musical experiments!
