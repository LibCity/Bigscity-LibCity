import time

from multiprocessing import cpu_count

from gensim.models import Word2Vec

from deepwalk import Node2VecWalk

class Node2Vec(Node2VecWalk, Word2Vec):
    """
    dimension: the dimension of embedded node
    window: window size for sampling in skipgram
    walks: random walk times for every node
    length: path length per walk
    lr: learning rate
    sampling: the method for sampling, 'negative' for negative sampling and 'huffman' for building Huffman Tree
    negative: number of negative sampling
    p, q: hyper-parameters that control the walk way, large p will cause the walk less likely to step back while large q
    will cause the walk step nearby the start node
    weight_reverse: whether reverse the weight, which means that larger edge weight will less likely to be visited.
    names: if the form of net is adjacent matrix, the names will be the label of all the nodes, the index is corresponded
    to the index in the matrix
    min_count: Though there provide the min_count to eliminate the nodes with litter influence (to eliminate the nodes
    that occur less than min_count times, this is not stable for this tools, because the min_count is clipping on the
    walk paths, which is varied in every trial. So it is better to set the min_count to 0.
    threads: multiprocessing option, choose the threads for parallel running, if == 0, threads = cpu_count - 1
    """
    def __init__(self, p=1, q=1, dimension=128, window=3, walks=10, length=10, weight_reverse=False,
                 lr=0.01, sampling='negative', negative=10, min_count=0, threads=0, names=None) -> object:
        self.__d = dimension
        self.__w = window
        self.__walks = walks
        self.__l = length
        self.__names = names
        if sampling == 'negative':
            if negative > 0:
                negative = negative
            else:
                negative = 10  # default value
        elif sampling == 'huffman':
            negative = 0
        else:
            raise ValueError('Only support "negative" and "huffman" option for the parameter "sampling".')
        if threads == 0:
            threads = cpu_count() - 1
        Node2VecWalk.__init__(self, p=p, q=q, weight_reverse=weight_reverse, walks=walks, length=length, names=names)
        Word2Vec.__init__(self, vector_size=dimension, window=window, alpha=lr, negative=negative, min_count=min_count, workers=threads)

    def trans_data(self, G, batch_size=1000):
        """
        :param G:
        :return:
        """
        allwalks = self.gen_walks(G, batch_size=batch_size)
        return allwalks

    def train_data(self, G, epochs=10, batch_size=1000, verbose_step=1):
        """
        epochs: training epochs
        verbose_step: the interval to print training information
        """
        degree = {node: deg for node, deg in G.degree}
        vocab = [node for node in degree.keys()]
        self.build_vocab_from_freq(degree)
        cumulate_time = 0
        for i in range(epochs):
            start = time.time()
            data = self.trans_data(G, batch_size=batch_size)
            losses = []
            for d in data:
                length_d = len(d)
                self.train(d, epochs=1, total_words=len(vocab), compute_loss=True)
                loss = (self.get_latest_training_loss(), length_d)
                losses.append(loss)
            ave_loss = sum([a for a, _ in losses]) / sum([b for _, b in losses])
            cumulate_time += time.time() - start
            if i < epochs - 1:
                if (i + 1) % verbose_step == 0:
                    print("{}/{} - ETA: {:.0f}s - loss: {:.4f}.".format(str(i + 1).rjust(len(str(epochs))), epochs,
                                                            cumulate_time / (i + 1) * (epochs - i - 1), ave_loss))
            else:
                print("{}/{} - Complete -loss: {:.4f}- Cost time: {:.0f}s.".format(str(i + 1).rjust(len(str(epochs))), epochs,
                                                                        ave_loss, cumulate_time))
        self.embedding = {node: self.wv[node] for node in vocab}