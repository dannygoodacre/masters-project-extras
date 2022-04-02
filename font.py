import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

mpl.rcParams.update(mpl.rcParamsDefault)
font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 12,})