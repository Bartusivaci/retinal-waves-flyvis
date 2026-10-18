import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

def generate_retinal_wave(
    size=64,
    num_frames=60,
    center=(-0.6, -0.4),
    max_radius=1.8,
    sigma=0.08,
):
    """
    Generate a simple expanding retinal-wave-like stimulus.

    Activity forms a ring-shaped wavefront that expands outward
    from a starting point over time.
    """

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    distance_from_center = np.sqrt(
        (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    )

    frames = []

    for t in range(num_frames):
        progress = t / (num_frames - 1)
        radius = progress * max_radius

        frame = np.exp(
            -((distance_from_center - radius) ** 2) / (2 * sigma**2)
        )

        frames.append(frame)

    return np.array(frames)


def save_animation(frames, output_path="results/retinal_wave.gif"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    image = ax.imshow(
        frames[0],
        cmap="gray",
        vmin=0,
        vmax=1,
        animated=True,
    )

    ax.set_title("Synthetic Retinal-Wave-Like Stimulus")
    ax.axis("off")

    def update(frame_index):
        image.set_array(frames[frame_index])
        return [image]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=50,
        blit=True,
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=20),
    )

    plt.close(fig)


if __name__ == "__main__":
    frames = generate_retinal_wave()

    print("Generated frames:", frames.shape)
    print("Minimum activity:", frames.min())
    print("Maximum activity:", frames.max())

    save_animation(frames)

    print("Saved animation to results/retinal_wave.gif")
