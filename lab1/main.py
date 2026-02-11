import python.monkdata as m
import python.dtree as d
import random
import matplotlib.pyplot as plt
import numpy as np
from python.drawtree_qt5 import drawTree
import matplotlib.lines as mlines

fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


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


def build_descion_tree():
    t1 = d.buildTree(m.monk1, m.attributes)
    t2 = d.buildTree(m.monk2, m.attributes)
    t3 = d.buildTree(m.monk3, m.attributes)

    print("Monk 1")
    print(f"Training data performance: {d.check(t1, m.monk1)}")
    print(f"Testing data performance: {d.check(t1, m.monk1test)}")
    print()

    print("Monk 2")
    print(f"Training data performance: {d.check(t2, m.monk2)}")
    print(f"Testing data performance: {d.check(t2, m.monk2test)}")
    print()

    print("Monk 3")
    print(f"Training data performance: {d.check(t3, m.monk3)}")
    print(f"Testing data performance: {d.check(t3, m.monk3test)}")
    print()


def build_tree_manual(monk_dataset):
    gains = []
    for attribute in m.attributes:
        data = d.averageGain(monk_dataset, attribute)
        gains.append(data)

    max_attribute = max(gains)
    a_index = gains.index(max_attribute)
    attribute = m.attributes[a_index]

    samples = []
    for i in attribute.values:
        ss = d.select(monk_dataset, attribute, i)
        samples.append(ss)

    for index, sample in enumerate(samples):
        gains = []
        for attribute in m.attributes:
            data = d.averageGain(sample, attribute)
            gains.append(data)

        max_attribute = max(gains)
        a_index = gains.index(max_attribute)
        attribute = m.attributes[a_index]
        print(f"Max for sample {index + 1}: attribute {a_index + 1}")
        print(f"Majority class for sample {index + 1}: {d.mostCommon(sample)}")


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune(tree, validation_set, depth):
    # print(f"Pruning for depth: {depth}")
    highest_acc = d.check(tree, validation_set)
    highest_tree = None
    for subtree in d.allPruned(tree):
        acc = d.check(subtree, validation_set)
        if acc > highest_acc:
            highest_tree = subtree
            highest_acc = acc

    if highest_tree is None:
        return tree

    depth += 1
    return prune(highest_tree, validation_set, depth)


def pruning(monk_dataset, monk_training):
    runs = 100
    means = []
    errors = []
    for f in fraction:
        temp = []
        for i in range(runs):
            monk_train, monk_val = partition(monk_dataset, f)
            t = d.buildTree(monk_train, m.attributes)
            highest_tree = prune(t, monk_val, 0)
            temp.append(d.check(highest_tree, monk_training))
        mean = np.mean(temp)
        var = np.var(temp)
        errors.append(var)
        means.append(mean)

    return means, errors


def boxplot_results(m1, e1, m2, e2):  # Added fraction to args for reproducibility

    plt.figure(figsize=(12, 6))
    plt.suptitle(
        "Model test performance with varying validation fractions (mean & var for 100 runs)",
        fontsize=16,
        y=0.98,
    )

    xlabel = "Training Fractions"
    ylabel = "Observed Accuracy"

    mean_proxy = mlines.Line2D(
        [], [], color="green", marker="o", linestyle="None", markersize=10, label="Mean"
    )
    var_proxy = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="|",
        linestyle="-",
        markersize=10,
        markeredgewidth=2,
        label="Variance",
    )

    plt.subplot(1, 2, 1)
    plt.errorbar(
        np.array(fraction),
        m1,
        yerr=e1,
        linestyle="None",
        marker="o",
        capsize=10,
        color="blue",
        mfc="green",
        mec="green",
    )
    plt.title("Monk 1 dataset")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="both", linestyle="--", alpha=0.7)

    plt.legend(handles=[mean_proxy, var_proxy], loc="upper left")

    plt.subplot(1, 2, 2)
    mean_proxy_sq = mlines.Line2D(
        [], [], color="green", marker="s", linestyle="None", markersize=10, label="Mean"
    )

    plt.errorbar(
        np.array(fraction),
        m2,
        yerr=e2,
        linestyle="None",
        marker="s",
        capsize=10,
        color="blue",
        mfc="green",
        mec="green",
    )
    plt.title("Monk 3 dataset")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="both", linestyle="--", alpha=0.7)

    plt.legend(handles=[mean_proxy_sq, var_proxy], loc="upper left")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()


def main():
    datasets = [m.monk1, m.monk2, m.monk3]

    m1, e1 = pruning(m.monk1, m.monk1test)
    m2, e2 = pruning(m.monk3, m.monk3test)

    tree = d.buildTree(m.monk1, m.attributes)
    drawTree(tree)

    boxplot_results(m1, e1, m2, e2)


if __name__ == "__main__":
    main()
