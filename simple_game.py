import os
import time
import random

LOG_FILE_ROW = 'bob_game/logs/game_row_log.txt'
LOG_FILE_GRID = 'logs/game_grid_log.txt'


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def log_row(row):
    with open(LOG_FILE_ROW, 'a') as file:
        file.write('|'.join(row) + '\n')


def log_grid(grid, file_path):
    with open(file_path, 'a') as file:
        for row in grid:
            file.write('|'.join(row) + '@')
        file.write('\n')


def main():
    total_loops = 500
    loop_atual = 0

    colunas = 3
    linhas = 5
    grid = [[' ' for _ in range(colunas)] for _ in range(linhas)]

    # open(LOG_FILE_ROW, 'w').close()
    open(LOG_FILE_GRID, 'w').close()

    while loop_atual < total_loops:
        clear_screen()



        joined_sublists = [''.join(sublist) for sublist in grid]

        # Set up initial barrier position
        if not '#' in ''.join(joined_sublists):
            barrier_col = random.randint(0, colunas - 1)
            grid[0][barrier_col] = '#'

        # log_row(grid[0])

        # Move barrier down
        grid = [[' ' for _ in range(colunas)]] + grid[:-1]

        grid[linhas - 1] = [colunas * "*"]

        log_grid(grid, LOG_FILE_GRID)
        # Draw grid
        for row in grid:
            print('|'.join(row))

        # print(10 * "#")
        # time.sleep(1)

        loop_atual += 1


if __name__ == "__main__":
    main()
