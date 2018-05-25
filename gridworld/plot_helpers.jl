using TikzPictures

function colorval(val, brightness::Real = 1.0; threshold = 0., val_scale = 5.)
    val = convert(Vector{Float64}, val)
    x = 255 - min.(255, 255 * (abs.(val) ./ val_scale) .^ brightness)
    r = 255 * ones(size(val))
    g = 255 * ones(size(val))
    b = 255 * ones(size(val))
    r[val .>= threshold] = x[val .>= threshold]
    b[val .>= threshold] = x[val .>= threshold]
    g[val .< threshold] = x[val .< threshold]
    b[val .< threshold] = x[val .< threshold]
    return (r, g, b)
end

function POMDPModels.plot(mdp::GridWorld, V::Vector, mask::SafetyMask{GridWorld, Symbol}, state=GridWorldState(0,0,true); threshold = 0.)
    o = IOBuffer()
    sqsize = 1.0
    twid = 0.05
    (r, g, b) = colorval(V, threshold=threshold)
    for s in iterator(states(mdp))
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval
            println(o, "\\definecolor{currentcolor}{RGB}{$(r[i]),$(g[i]),$(b[i])}")
            println(o, "\\fill[currentcolor] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            if s == state
                println(o, "\\fill[orange] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            end
        end
    end
    println(o, "\\begin{scope}[fill=gray]")
    for s in iterator(states(mdp))
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval + 1
            c = [xval, yval] * sqsize - sqsize / 2
            C = [c'; c'; c']'
            RightArrow = [0 0 sqsize/2; twid -twid 0]
            actions = safe_actions(mask, s)
            for dir in actions
                if dir == :left
                    A = [-1 0; 0 -1] * RightArrow + C
                    println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
                end
                if dir == :right
                    A = RightArrow + C
                    println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
                end
                if dir == :up
                    A = [0 -1; 1 0] * RightArrow + C
                    println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
                end
                if dir == :down
                    A = [0 1; -1 0] * RightArrow + C
                    println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
                end
            end

            vs = @sprintf("%0.2f", V[i])
            println(o, "\\node[above right] at ($((xval-1) * sqsize), $((yval-1) * sqsize)) {\$$(vs)\$};")
        end
    end
    println(o, "\\end{scope}");
    println(o, "\\draw[black] grid(10,10);");
    TikzPicture(String(take!(o)), options="scale=1.25")
end

function POMDPModels.plot(mdp::GridWorld, V::Vector, policy::Policy, state=GridWorldState(0,0,true); threshold::Float64 = 0.)
    o = IOBuffer()
    sqsize = 1.0
    twid = 0.05
    (r, g, b) = colorval(V, threshold=threshold)
    for s in iterator(states(mdp))
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval
            println(o, "\\definecolor{currentcolor}{RGB}{$(r[i]),$(g[i]),$(b[i])}")
            println(o, "\\fill[currentcolor] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            if s == state
                println(o, "\\fill[orange] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            end
        end
    end
    println(o, "\\begin{scope}[fill=gray]")
    for s in iterator(states(mdp))
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval + 1
            c = [xval, yval] * sqsize - sqsize / 2
            C = [c'; c'; c']'
            RightArrow = [0 0 sqsize/2; twid -twid 0]
            dir = action(policy, s)
            if dir == :left
                A = [-1 0; 0 -1] * RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end
            if dir == :right
                A = RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end
            if dir == :up
                A = [0 -1; 1 0] * RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end
            if dir == :down
                A = [0 1; -1 0] * RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end

            vs = @sprintf("%0.2f", V[i])
            println(o, "\\node[above right] at ($((xval-1) * sqsize), $((yval-1) * sqsize)) {\$$(vs)\$};")
        end
    end
    println(o, "\\end{scope}");
    println(o, "\\draw[black] grid(10,10);");
    TikzPicture(String(take!(o)), options="scale=1.25")
end
