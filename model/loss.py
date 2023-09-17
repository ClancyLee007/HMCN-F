import tensorflow as tf
from meta.tag_tree import TagTree

class Hierarchical_Violation_Loss(tf.keras.losses.Loss):

    def __init__(self, alpha=0.1, name="hierarchical_violation_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tagTree = TagTree()
        self.alpha = alpha
        self.initFile = "D://project/hmcn/meta/tag_content_tree_map.txt"
        self.ip_idx = []
        self.solve_father_id_list()

    def call(self, y_true, y_pred):
        y_ip = tf.gather(tf.pad(y_pred, [[0, 0], [1, 0]]), tf.constant(self.ip_idx), axis=-1)
        temp = tf.nn.relu(y_pred - y_ip)
        return tf.linalg.diag_part(self.alpha * tf.matmul(temp, tf.transpose(temp)))

    def solve_father_id_list(self):
        with open(self.initFile, encoding='utf8') as f:
            for line in f:
                pair = line.strip("\n").split("\t")
                self.ip_idx.append(self.tagTree.tag_id_map.get(pair[1]) if pair[0] == "无" else self.tagTree.tag_id_map.get(pair[0]))
