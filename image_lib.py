import random
def shuffle(origin_list):
    result_list = []
    for i in range(len(origin_list)):
        index = random.randint(0, len(origin_list)-1)
        result_list.append(origin_list[index])
        del origin_list[index]
    return result_list