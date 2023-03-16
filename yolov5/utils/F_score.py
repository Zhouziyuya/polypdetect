
def f1(precision, recall):
    f1_score = (2*precision*recall)/(precision+recall)
    return f1_score


def f2(precision, recall):
    f2_score = (5*precision*recall)/(4*precision+recall)
    return f2_score


if __name__ == "__main__":
    a = f1(0.229, 0.279)
    print(a)