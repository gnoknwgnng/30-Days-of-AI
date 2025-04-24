import numpy as np
from music21 import stream, note, midi
import os

output_dir = "./generated_midi"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters for melody generation
num_notes = 32
scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
octave = 4

# Generate random melody
melody = stream.Stream()
for _ in range(num_notes):
    pitch = np.random.choice(scale)
    n = note.Note(f"{pitch}{octave}")
    n.quarterLength = np.random.choice([0.25, 0.5, 1.0])  # random duration
    melody.append(n)

# Save as MIDI file
output_file = os.path.join(output_dir, "generated_melody.mid")
mf = midi.translate.streamToMidiFile(melody)
mf.open(output_file, 'wb')
mf.write()
mf.close()

print(f"Check your MIDI at {output_file}")