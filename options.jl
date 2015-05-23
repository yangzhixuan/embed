# The types defined below are used for specifying the options of the word embedding training
abstract Option

################## initialization methods ########################
abstract InitializatioinMethod <: Option
type RandomInited <: InitializatioinMethod 
    # Initialize the embedding randomly
end
random_inited = RandomInited()

################### network structure ############################
abstract NetworkType <: Option
type NaiveSoftmax <: NetworkType
    # |V| outputs softmax
end
type HuffmanTree <: NetworkType
    # Predicate step by step on the huffman tree
end
naive_softmax = NaiveSoftmax()
huffman_tree = HuffmanTree()
