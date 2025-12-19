import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import subprocess
import random
from flask import Flask, render_template, request, send_file, jsonify
from music21 import converter, note, chord, stream
from pydub import AudioSegment
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

L, D = 100, "cpu"

# Load vocabulary and mappings
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)
vocab = sorted(set(notes))
n_to_i = {n: i for i, n in enumerate(vocab)}
i_to_n = {i: n for i, n in enumerate(vocab)}
v_size = len(vocab)

# Define and load model
E = nn.Embedding(v_size, 256)
T = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, batch_first=True), 4)
F = nn.Linear(256, v_size)

sd = torch.load("./model/epoch_15.pth", map_location=D)
E.load_state_dict({k.split('.')[-1]: v for k, v in sd.items() if 'embed' in k})
T.load_state_dict({k.replace('transformer.', ''): v for k, v in sd.items() if 'transformer' in k})
F.load_state_dict({k.split('.')[-1]: v for k, v in sd.items() if 'fc' in k})
for m in (E, T, F):
    m.eval()


def convert_mp3_to_midi(mp3_path, output_midi_path):
    try:
        wav_path = mp3_path.replace('.mp3', '_temp.wav')
        AudioSegment.from_mp3(mp3_path).export(wav_path, format='wav')
        
        from basic_pitch.inference import predict
        
        model_output, midi_data, note_events = predict(wav_path)
        
        import pretty_midi
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        for start_time, end_time, pitch, velocity, _ in note_events:
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=float(start_time),
                end=float(end_time)
            )
            instrument.notes.append(note)
        
        pm.instruments.append(instrument)
        pm.write(output_midi_path)
        
        if os.path.exists(wav_path):
            os.remove(wav_path)
        
        return True
    except Exception as e:
        print(f"MP3 to MIDI conversion error: {e}")
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        return False


def generate_music(midi_path, num_notes):
    try:
        m = converter.parse(midi_path)
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI file: {e}")

    # Parse notes and chords using MIDI pitch numbers (consistent with training data)
    ext = []
    for el in m.recurse():
        if isinstance(el, note.Note):
            ext.append(str(el.pitch.midi))
        elif isinstance(el, chord.Chord):
            pitches = [str(n.pitch.midi) for n in el.notes]
            ext.append('.'.join(pitches))
    
    # Filter to only known tokens
    ext = [n for n in ext if n in n_to_i]

    # Fallback if no valid notes found
    if not ext:
        print("Warning: No notes from input MIDI found in vocabulary. Using random fallback.")
        ext = [random.choice(list(n_to_i.keys()))]

    # Ensure at least L notes by repeating
    if len(ext) < L:
        ext = (ext * ((L // len(ext)) + 1))[:L]
    pat = [n_to_i[n] for n in ext[:L]]

    res = []
    for _ in range(num_notes):
        x = torch.tensor([pat], dtype=torch.long)
        with torch.no_grad():
            embedded = E(x)                     # [1, L, 256]
            transformed = T(embedded)           # [1, L, 256]
            logits = F(transformed[:, -1, :])   # [1, v_size]
            probs = torch.softmax(logits, dim=1).numpy()[0]
        idx = np.random.choice(len(probs), p=probs)
        res.append(i_to_n[idx])
        pat = (pat + [idx])[1:]  # sliding window

    # Convert generated tokens back to music21 objects
    output_notes = []
    for i, el in enumerate(res):
        if '.' in el:
            # Chord: split and convert to Note objects via MIDI numbers
            try:
                pitches = [int(p) for p in el.split('.')]
                c = chord.Chord(pitches)
                c.offset = i * 0.5
                output_notes.append(c)
            except:
                # Fallback to single note if invalid
                n = note.Note(int(el.split('.')[0]))
                n.offset = i * 0.5
                output_notes.append(n)
        else:
            # Single note
            try:
                n = note.Note(int(el))
                n.offset = i * 0.5
                output_notes.append(n)
            except:
                continue  # skip invalid

    if not output_notes:
        raise ValueError("No valid notes generated.")

    return output_notes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    if 'midi_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['midi_file']
    num_notes = int(request.form.get('num_notes', 300))

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not any(file.filename.lower().endswith(ext) for ext in ['.mid', '.midi', '.mp3']):
        return jsonify({'error': 'Only MIDI and MP3 files allowed'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    midi_path = None
    try:
        if filename.lower().endswith('.mp3'):
            midi_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + '.mid')
            success = convert_mp3_to_midi(input_path, midi_path)
            if not success:
                return jsonify({'error': 'Failed to convert MP3 to MIDI'}), 500
            os.remove(input_path)
        else:
            midi_path = input_path

        output_notes = generate_music(midi_path, num_notes)

        output_id = os.path.splitext(filename)[0] + '_generated'
        mid_p = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_id}.mid")
        wav_p = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_id}.wav")
        mp3_p = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_id}.mp3")

        stream.Stream(output_notes).write("midi", mid_p)

        if not os.path.exists("font.sf2"):
            return jsonify({'error': 'font.sf2 not found'}), 500

        subprocess.run([
            "fluidsynth", "-ni", "-g", "1", "-F", wav_p, "-r", "44100", "font.sf2", mid_p
        ], check=True, capture_output=True)

        AudioSegment.from_wav(wav_p).export(mp3_p, format="mp3")

        # Cleanup
        for p in [wav_p, mid_p, midi_path]:
            if p and os.path.exists(p):
                os.remove(p)

        return jsonify({
            'success': True,
            'filename': f"{output_id}.mp3"
        })

    except Exception as e:
        # Ensure cleanup on error
        for p in [input_path, midi_path]:
            if p and os.path.exists(p):
                os.remove(p)
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )


@app.route('/play/<filename>')
def play(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        mimetype='audio/mpeg'
    )


if __name__ == '__main__':
    app.run(debug=True)