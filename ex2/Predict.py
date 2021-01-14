def preHelp(des_tree, test_list, label_dic):
    for e in des_tree.keys():
        best_feature = e
    for e in test_list:
        if label_dic[e] == best_feature:
            best = des_tree[best_feature][e]
            if type(best) == type({}):
                return preHelp(best, test_list, label_dic)
            else:
                return best


def predict(des_tree, test_data, label_dic, real_target):
    pre_list = []
    for e in test_data:
        pre_list.append(preHelp(des_tree, e, label_dic))
    right_count = 0
    for i in range(len(pre_list)):
        if pre_list[i] == real_target[i]:
            right_count += 1
    right_rate = right_count / len(pre_list)
    return pre_list, right_rate


if __name__ == "__main__":
    Tree = {
        '纹理': {
            '清晰': {'根蒂': {
                '硬挺': '坏瓜',
                '蜷缩': '好瓜',
                '稍蜷': {
                    '色泽': {
                        '青绿': '好瓜',
                        '乌黑': {
                            '触感': {
                                '软粘': '坏瓜',
                                '硬滑': '好瓜'}}}}}},
            '模糊': '坏瓜',
            '稍糊': {
                '触感': {
                    '硬滑': '坏瓜',
                    '软粘': '好瓜'}}}}
    test_data = [['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
                 ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
                 ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
                 ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘']]
    ideal_target = ['好瓜', '好瓜', '坏瓜', '坏瓜']
    labels = {'青绿': '色泽', '乌黑': '色泽', '浅白': '色泽', '蜷缩': '根蒂', '稍蜷': '根蒂', '硬挺': '根蒂', '清脆': '敲声', '浊响': '敲声',
              '沉闷': '敲声',
              '清晰': '纹理', '模糊': '纹理', '稍糊': '纹理', '凹陷': '脐部', '平坦': '脐部', '稍凹': '脐部', '软粘': '触感', '硬滑': '触感'}
    print(predict(Tree, test_data, labels, ideal_target))
