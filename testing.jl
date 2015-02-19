require("utils.jl")
require("word_embedding.jl")
require("softmax_classifier.jl")
using IProfile

function test_word_window()
    fp = open("test_text")
    for i in words_of(fp)
        println(i)
        println("--------")
    end

    seekstart(fp)
    for i in imap(uppercase, words_of(fp))
        println(i)
    end

    seekstart(fp)
    for i in sliding_window(words_of(fp))
        println(i)
    end
end

function test_softmax()
    @printf "Testing the softmax classifier on the MNIST dataset\n"
    @printf "Loading...\n"
    D = readcsv("mnist_train.csv")
    X_train = D[:, 2:end] / 255
    y_train = int64(D[:, 1] + 1)

    D = readcsv("mnist_test.csv")
    X_test = D[:, 2:end] / 255
    y_test= int64(D[:, 1] + 1)

    @printf "Start training...\n"
    c = LinearClassifier(10, 784)
    @time train_parallel(c, X_train, y_train, max_iter = 50)
    @printf "Accuracy on test set %f (a value around 0.9 is expected)\n" accuracy(c, X_test, y_test)
end

function test_word_embedding()
    embed = WordEmbedding(30, random_inited, huffman_tree, subsampling = 0)
    @time train(embed, "text8_tiny")
    embed
end

function test_word_embedding_large()
    embed = WordEmbedding(30, random_inited, huffman_tree, subsampling = 1e-4)
    @time train(embed, "text8")
    embed
end
