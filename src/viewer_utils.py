import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_pc(pc, seg=None, background=False, unit=False, save=None):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    if not background:
        ax.grid(False)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    else:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    if seg is not None:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=seg, cmap='viridis', s=2)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], cmap='viridis', s=2)
    if unit:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    if save is not None:
        plt.savefig(save)


def draw_field(pc, field, seg=None, background=False, unit=False, save=None):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    if not background:
        ax.grid(False)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    else:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    if seg is not None:
        ax.quiver(pc[0, 0, :], pc[0, 1, :], pc[0, 2, :],
                  field[0, 0, :], field[0, 1, :], field[0, 2, :],
                  color=seg, cmap='viridis')
    else:
        ax.quiver(pc[0, 0, :], pc[0, 1, :], pc[0, 2, :],
                  field[0, 0, :], field[0, 1, :], field[0, 2, :],
                  cmap='viridis')
    if unit:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    if save is not None:
        plt.savefig(save)


def draw_grid(grid, neighbors, dim=3, pc=None, seg=None):
    """
    :param grid: (num_points, channel)
    """
    plt.figure()
    if dim == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(grid.shape[0]):
            for j in neighbors[i]:
                ax.plot((grid[i, 0], grid[j, 0]),
                        (grid[i, 1], grid[j, 1]),
                        (grid[i, 2], grid[j, 2]))
        if pc is not None:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=seg, s=10)
    elif dim == 2:
        for i in range(grid.shape[0]):
            for j in neighbors[i]:
                plt.plot((grid[i, 0], grid[j, 0]),
                         (grid[i, 1], grid[j, 1]))
        if pc is not None:
            print(pc.shape)
            plt.scatter(pc[:, 0], pc[:, 1], c=seg, s=10)
