import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

# Function to draw a labeled box
def draw_box(ax, xy, width, height, text, color="lightblue"):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", fc=color, ec="black", lw=1.5)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10, weight="bold")

# Draw ViViT encoder
draw_box(ax, (1, 6), 2, 1, "Input Video\nFrames", color="lightgrey")
draw_box(ax, (4, 6), 3, 1, "Spatial Transformer\nEncoder")
draw_box(ax, (8, 6), 3, 1, "Temporal Transformer\nEncoder")
draw_box(ax, (12, 6), 2, 1, "Embed to Tokens", color="lightgrey")

# Arrows for ViViT
ax.annotate("", xy=(3, 6.5), xytext=(1.9, 6.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(7, 6.5), xytext=(6, 6.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(11, 6.5), xytext=(9, 6.5), arrowprops=dict(arrowstyle="->", lw=1.5))

# Draw GPT-4 decoder
draw_box(ax, (6, 3), 4, 1, "GPT-4 via\nOpenAI API", color="lightgreen")
draw_box(ax, (6, 1), 4, 1, "Generated\nCaptions", color="lightgrey")

# Arrows for GPT-4
ax.annotate("", xy=(8, 3.5), xytext=(8, 6), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(8, 1.5), xytext=(8, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

# Titles and labels
ax.text(6, 8, "ViViT + GPT-4 Integration for Video Captioning", ha="center", va="center", fontsize=14, weight="bold")

plt.show()
