import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as ex

import os

if not os.path.exists('graphs'):
    os.makedirs('graphs')

data_folder = "data"
languages = next(os.walk(data_folder))[1]

language_files = {}
language_sizes = {}

for language in languages:
    language_path = os.path.join(data_folder, language)
    _, _, files = next(os.walk(language_path))
    language_files[language] = len(files)
    file_sizes = [os.path.getsize(os.path.join(language_path, file)) for file in files]
    language_sizes[language] = sum(file_sizes)


df = pd.DataFrame(list(language_files.items()), columns=['Language', 'Files'])

fig = ex.bar(df, x='Language', y='Files', title='Number of files per language')
fig.show()
fig.write_image("graphs/files_per_language.png")

# plot total number of files with amount of different languages
total_files = sum(language_files.values())
languages = list(language_files.keys())
files = list(language_files.values())
fig = ex.pie(df, values=files, names=languages, title='Number of files per language', hole=0.3)
fig.show()
fig.write_image("graphs/files_per_language_pie.png")

# plot file size per language

df = pd.DataFrame(list(language_sizes.items()), columns=['Language', 'Size'])

# plot file size per language
fig = ex.bar(df, x='Language', y='Size', title='Size of files per language')
fig.show()
fig.write_image("graphs/file_size_per_language.png")

# plot amount of lines per language
language_lines = {}

for language in languages:
    language_path = os.path.join(data_folder, language)
    _, _, files = next(os.walk(language_path))
    file_lines = []
    for file in files:
        with open(os.path.join(language_path, file), 'r', errors="ignore") as f:  # ignore encoding errors
            file_lines.append(len(f.readlines()))
    language_lines[language] = sum(file_lines)

df = pd.DataFrame(list(language_lines.items()), columns=['Language', 'Lines'])

# plot file lines per language
fig = ex.bar(df, x='Language', y='Lines', title='Number of lines per language')
fig.show()
fig.write_image("graphs/lines_per_language.png")


