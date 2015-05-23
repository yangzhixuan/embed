using Iterators
using LightXML

type WordStream
    fp :: Union(IOStream, String)
    startpoint :: Int64
    endpoint :: Int64
    buffer :: IOBuffer

    # filter configuration
    rate :: Float64   # if rate > 0, words will be subsampled according to distr
    filter :: Bool    # if filter is true, only words present in the keys(distr) will be considered
    distr :: Dict{String, Float64}
end

function words_of(file :: Union(IOStream, String); subsampling = (0, false, nothing), startpoint = -1, endpoint = -1)
    rate, filter, distr = subsampling
    WordStream(file, startpoint, endpoint, IOBuffer(), rate, filter, rate == 0 && !filter ? Dict() : distr)
end

function parallel_words_of(filename :: String, num_workers :: Integer; subsampling = (0, false, nothing))
     fp = open(filename, "r")
     seekend(fp)
     flen = position(fp)
     close(fp)
 
     per_len = int(floor(flen / num_workers))
     cursor = 0
     res = cell(num_workers)
     for i in 1:num_workers
         last = (i == num_workers ? flen - 1 : cursor + per_len - 1)
         res[i] = words_of(filename, subsampling = subsampling, startpoint = cursor, endpoint = last)
         cursor += per_len
     end
     res
end

function Base.start(ws :: WordStream)
    if isa(ws.fp, String)
        ws.fp = open(ws.fp)
    end
    if ws.startpoint >= 0
        seek(ws.fp, ws.startpoint)
    else
        ws.startpoint = 0
        seekend(ws.fp)
        ws.endpoint = position(ws.fp)
        seekstart(ws.fp)
    end
    nothing
end

function Base.done(ws :: WordStream, state)
    while !eof(ws.fp)
        if ws.endpoint >= 0 && position(ws.fp) > ws.endpoint
            break
        end
        c = read(ws.fp, Char)
        if c == ' ' || c == '\n' || c == '\0' || c == '\r'
            s = takebuf_string(ws.buffer)
            if s == "" || (ws.filter && !haskey(ws.distr, s))
                continue
            end
            if ws.rate > 0
                prob = (sqrt(ws.distr[s] / ws.rate) + 1) * ws.rate / ws.distr[s]
                if(prob < rand())
                    # @printf "throw %s, prob is %f\n" s prob
                    continue;
                end
            end
            write(ws.buffer, s)
            return false
        else
            write(ws.buffer, c)
        end
    end
    #close(ws.fp)
    return true
end

function Base.next(ws :: WordStream, state)
    (takebuf_string(ws.buffer), nothing)
end

type SlidingWindow
    ws :: WordStream
    lsize :: Int64
    rsize :: Int64
end

function Base.start(window :: SlidingWindow)
    convert(Array{String, 1}, collect(take(window.ws, window.lsize + 1 + window.rsize)))
end

function Base.done(window :: SlidingWindow, w :: Array{String})
    done(window.ws, nothing)
end

function Base.next(window :: SlidingWindow, w :: Array{String})
    shift!(w)
    push!(w, next(window.ws, nothing)[1])
    (w, w)
end

function sliding_window(words; lsize = 5, rsize = 5)
    SlidingWindow(words, lsize, rsize)
end

abstract TreeNode
type BranchNode <: TreeNode
    children :: Array{BranchNode, 1}
    data
    extrainfo
end
type NullNode <: TreeNode
end
nullnode = NullNode()

function leaves_of(root :: TreeNode)
    code = Int64[]
    function traverse(node :: TreeNode)
        if node == nullnode
            return
        end
        if length(node.children) == 0
            produce((node.data, copy(code)))    # notice that we should copy the current state of code
        end
        for (index, child) in enumerate(node.children)
            push!(code, index)
            traverse(child)
            pop!(code)
        end
    end
    Task(() -> traverse(root))
end

function internal_nodes_of(root :: TreeNode)
    function traverse(node :: TreeNode)
        if node == nullnode
            return
        end
        if length(node.children) != 0
            produce(node)
        end
        for child in node.children
            traverse(child)
        end
    end
    Task(() -> traverse(root))
end

function partition{T}(a :: Array{T}, n :: Integer)
    b = Array{T}[]
    t = int(floor(length(a) / n))
    cursor = 1
    for i in 1:n
        push!(b, a[cursor : (i == n ? length(a) : cursor + t - 1)])
        cursor += t
    end
    b
end

function read_word2vec_txt(fname :: String)
    lines = readlines(open(fname))
    dim = int(split(lines[1], " ")[2])
    emb = WordEmbedding(dim, random_inited, huffman_tree)

    for l in lines[2:end]
        splitted = split(strip(l), " ")
        a = zeros(1, dim)
        for (ind, field) in enumerate(splitted[2:end])
            try 
                a[ind] = float(field)
            catch err
                @printf "invalid float: %s\n" field
            end
        end
        emb.embedding[splitted[1]] = a
    end
    emb
end


function average_height(tree :: TreeNode)
    (h, c) = (0, 0)
    for (_, path) in leaves_of(tree)
        h += length(path)
        c += 1
    end
    h / c
end
