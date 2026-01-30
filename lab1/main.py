import python.monkdata as m
import python.dtree as d


def calculate_attribute_entropy(datasets):
    print("| Dataset | a1 | a2 | a3 | a4 | a5 | a6 |")
    print("|---------|----|----|----|----|----|----|")

    monk_attribute_gains = []

    for monk_index, monk in enumerate(datasets):
        print(f"| Monk-{monk_index + 1} ", end=" ")
        monk_attributes_gain = []
        for i, attribute in enumerate(m.attributes):
            gain = d.averageGain(monk, attribute)
            monk_attributes_gain.append(gain)

            print(f" | {round(d.averageGain(monk, attribute), 4)}", end="")
        print(" |")
        monk_attribute_gains.append(monk_attributes_gain)

    print()

    for i in range(len(monk_attribute_gains)):
        min_attribute = max(monk_attribute_gains[i])
        a_index = monk_attribute_gains[i].index(min_attribute)
        print(
            f"Monk {i + 1}: Max attribute {a_index} gain: {round(max(monk_attribute_gains[i]), 4)}"
        )


def calculate_monk_entropy():
    entropy_m1 = d.entropy(m.monk1)
    entropy_m2 = d.entropy(m.monk2)
    entropy_m3 = d.entropy(m.monk3)

    return entropy_m1, entropy_m2, entropy_m3


def main():
    datasets = [m.monk1, m.monk2, m.monk3]

    calculate_attribute_entropy(datasets)


if __name__ == "__main__":
    main()
