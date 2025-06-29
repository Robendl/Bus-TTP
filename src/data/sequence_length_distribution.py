import matplotlib.pyplot as plt

def plot_seq_length_distribution(df_route):
    sequence_lengths = df_route.groupby("route_seq_hash").size()
    print(sum(sequence_lengths > 90))

    # Plot de distributie
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50)
    plt.title("Distributie van sequence lengtes per route_seq_hash")
    plt.xlabel("Sequence lengte (aantal wegvakken)")
    plt.ylabel("Aantal routes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()