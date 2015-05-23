require("utils.jl")
require("softmax_classifier.jl")
require("options.jl")

using Base.Collections      # for priority queue
using IProfile
using Distances

type WordEmbedding
    vocabulary :: Array{String}
    embedding :: Dict{String, Array{Float64}}
    classification_tree :: TreeNode
    distribution :: Dict{String, Float64}
    codebook :: Dict{String, Vector{Int64}}

    init_type :: InitializatioinMethod
    network_type :: NetworkType
    dimension :: Int64
    lsize :: Int64    # left window size in training
    rsize :: Int64    # right window size
    trained_count :: Int64
    corpus_size :: Int64
    subsampling :: Float64
    init_learning_rate :: Float64
    iter :: Int64
    min_count :: Int64
end

function WordEmbedding(dim :: Int64, init_type :: InitializatioinMethod, network_type :: NetworkType; lsize = 5, rsize = 5, subsampling = 1e-5, init_learning_rate = 0.025, iter = 5, min_count = 5)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
    return WordEmbedding(String[], Dict(), nullnode, Dict(), Dict(), init_type, 
        network_type, dim, lsize, rsize, 0, 0, subsampling, init_learning_rate, iter, min_count);
end

# now we can start the training
function work_process(embed :: WordEmbedding, words_stream :: WordStream,  number_workers :: Int64)
    middle = embed.lsize + 1
    input_gradient = zeros(Float64, embed.dimension)
    α = embed.init_learning_rate
    trained_count = 0
    flen = words_stream.endpoint - words_stream.startpoint
    trained_times = Dict{String, Int64}()

    @printf "start training...\n"
    for current_iter in 1:embed.iter
        for window in sliding_window(words_stream, lsize = embed.lsize, rsize = embed.rsize)
            trained_word = window[middle]
            trained_times[trained_word] = get(trained_times, trained_word, 0) + 1
            trained_count += 1
            if trained_count % 10000 == 0
                iter_progress = (position(words_stream.fp) - words_stream.startpoint) / flen
                progress = ((current_iter - 1) + iter_progress) / embed.iter
                @printf "trained on %d words, progress %.2f, α = %f\n" trained_count progress α
                α = embed.init_learning_rate * (1 - progress)
                if α < embed.init_learning_rate * 0.0001
                    α = embed.init_learning_rate * 0.0001
                end
            end
            local_lsize = int(rand(Uint64) % embed.lsize)
            local_rsize = int(rand(Uint64) % embed.rsize)
            # @printf "lsize: %d, rsize %d\n" local_lsize local_rsize
            for ind in middle - local_lsize : middle + local_rsize
                if ind == middle
                    continue
                end
                target_word = window[ind]

                if !haskey(embed.codebook, target_word)
                    # discard words not presenting in the classification tree
                    continue;
                end
                # @printf "%s -> %s\n" trained_word target_word
                node = embed.classification_tree :: TreeNode
                fill!(input_gradient, 0.0)
                input = embed.embedding[trained_word]

                for code in embed.codebook[target_word]
                    train_one(node.data, input, code, input_gradient, α)
                    node = node.children[code]
                end

                for i in 1:embed.dimension
                    input[i] -= input_gradient[i]
                end
            end
        end
    end
    @printf "finished training, sending result to the main process\n"
    (embed.embedding, trained_count, trained_times)
end

function train(embed :: WordEmbedding, corpus_filename :: String)
    fs = open(corpus_filename, "r")
    word_count = 0
    @printf "scanning the file...\n"
    for i in words_of(fs)
        if !haskey(embed.distribution, i)
            embed.distribution[i] = 0
        end
        embed.distribution[i] += 1
        word_count += 1
        if word_count % 1000000 == 0
            @printf "%d million words are read\n" (word_count // 1000000)
        end
    end
    for (k, v) in embed.distribution
        if v < embed.min_count
            delete!(embed.distribution, k)
            continue
        end
        embed.distribution[k] = v / word_count
    end

    embed.corpus_size = word_count
    embed.vocabulary = collect(keys(embed.distribution))

    @printf "corpus size: %d words\n" word_count
    @printf "vocabulary size: %d\n" length(embed.vocabulary)

    initialize_embedding(embed, embed.init_type)        # initialize by the specified method
    initialize_network(embed, embed.network_type)

    # determine the position in the tree for every word
    for (w, code) in leaves_of(embed.classification_tree)
        embed.codebook[w] = code
    end

    function reduce_embeds!(embed :: WordEmbedding, rets)
        for word in embed.vocabulary
            total_trained_times = sum(map(ret->get(ret[3], word, 0), rets))
            embed.embedding[word] = sum(map(ret -> get(ret[3], word, 0) * ret[1][word], rets)) / total_trained_times
        end
        embed.trained_count += sum(map(ret->ret[2], rets))
        embed
    end

    number_workers = nworkers()
    @printf "distribute work to %d processes\n" number_workers
    words_streams = parallel_words_of(corpus_filename, number_workers, subsampling = (embed.subsampling, true, embed.distribution))
    tic()
    reduce_embeds!(embed, pmap(work_process, [embed for i in 1:number_workers], words_streams, [number_workers for j in 1:number_workers]))
    time_used = toq()
    @printf "training finished in %d seconds (speed: %d words/sec)\n" time_used (word_count * embed.iter / time_used)
    embed
end

function initialize_embedding(embed :: WordEmbedding, randomly :: RandomInited)
    for i in embed.vocabulary
        embed.embedding[i] = rand(1, embed.dimension) * 2 - 1
    end
    embed
end

function build_huffman_tree(distr :: Dict{String, Float64}; func :: Function = ((node1,node2)->nothing))
    heap = PriorityQueue()
    for (word, freq) in distr
        node = BranchNode([], word, nothing)    # the data field of leaf node is its corresponding word.
        enqueue!(heap, node, freq)
    end
    while length(heap) > 1
        (node1, freq1) = peek(heap)
        dequeue!(heap)
        (node2, freq2) = peek(heap)
        dequeue!(heap)
        newnode = BranchNode([node1, node2], func(node1, node2), nothing) # the data field of internal node is the classifier
        enqueue!(heap, newnode, freq1 + freq2)
    end
    dequeue!(heap)
end

function initialize_network(embed :: WordEmbedding, huffman :: HuffmanTree)
    tree = build_huffman_tree(embed.distribution, func = (a,b)->(LinearClassifier(2, embed.dimension)))
    embed.classification_tree = tree
    embed
end

function Base.show(io :: IO, x :: WordEmbedding)
    @printf io "Word embedding(dimension = %d) of %d words, trained on %d words\n" x.dimension length(x.vocabulary) x.trained_count
    for (word, embed) in take(x.embedding, 5)
        @printf io "%s => %s\n" word string(embed)
    end
    if length(x.embedding) > 5
        @printf io "......"
    end
end


##################### codes for experiements ####################################

function find_nearest_words(embed :: WordEmbedding, word_embed :: Array{Float64}; k = 5)
    pq = PriorityQueue(Base.Order.Reverse)

    for (w, embed_w) in embed.embedding
        enqueue!(pq, w, cosine_dist(vec(word_embed), vec(embed_w)))
        if length(pq) > k
            dequeue!(pq)
        end
    end
    sort(collect(pq), by = t -> t[2])

end

function find_nearest_words(embed :: WordEmbedding, word :: String; k = 5)
    if !haskey(embed.embedding, word)
        msg = @sprintf "'%s' doesn't present in the embedding\n" word
        warn(msg)
        return nothing
    end
    find_nearest_words(embed, embed.embedding[word], k = k)
end

function similarity_test(embed :: WordEmbedding, candidates :: Set{String}, test_pairs :: Array{(String, String)})
    mri = 0
    count = 0
    for (a, b) in test_pairs
        dis = sqeuclidean(vec(embed.embedding[a]), vec(embed.embedding[b]))
        rank = 0
        for c in candidates
            if sqeuclidean(vec(embed.embedding[a]), vec(embed.embedding[c])) <= dis
                rank += 1
            end
        end
        if rank == 0
            rank = 1
        end
        mri += 1/rank
        count += 1
    end
    mri / count
end
