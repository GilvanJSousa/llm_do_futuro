def read_and_create_variable(file_path):
    unique_lines = set()

    try:
        with open(file_path, 'r') as file:
            for line in file:
                unique_lines.add(line.strip())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    return list(unique_lines)

def main():
    file_path = 'logs/game_grid_log.txt'  # Replace with the path to your text file
    unique_lines = read_and_create_variable(file_path)

    if unique_lines is not None:
        print("Unique lines:")
        for index, line in enumerate(unique_lines):
            print(index, line)
    print("Tamanho", len(unique_lines))

    stoi = {ch: i for i, ch in enumerate(unique_lines)}
    itos = {i: ch for i, ch in enumerate(unique_lines)}
    encode = lambda s: [stoi[c[1:-1]] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    print(stoi)
    print(itos)

    with open(file_path, 'r', encoding='utf-8') as f:
        # text = f.read()
        text = f.readlines()
    encode(text)

if __name__ == "__main__":
    main()