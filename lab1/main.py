import python.monkdata as m
import python.dtree as d
from python.drawtree_qt5 import drawTree


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
    
def pruning():
    pass

def build_tree_manual(monk_dataset):
    gains = []
    for attribute in m.attributes:
        data = d.averageGain(monk_dataset, attribute)
        gains.append(data)
    
    max_attribute = max(gains)
    a_index = gains.index(max_attribute)
    print("attr :", a_index)
    
    drawTree(monk_dataset)
    t = d.select(monk_dataset, m.attributes[a_index], max_attribute)
    drawTree(monk_dataset)
    print(d.mostCommon(monk_dataset))
    print(t)

def main():
    datasets = [m.monk1, m.monk2, m.monk3]
    
    build_tree_manual(m.monk1)
    

if __name__ == "__main__":
    main()
