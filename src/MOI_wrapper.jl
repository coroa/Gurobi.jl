import MathOptInterface

const MOI = MathOptInterface
const CleverDicts = MOI.Utilities.CleverDicts

@enum(VariableType, CONTINUOUS, BINARY, INTEGER, SEMIINTEGER, SEMICONTINUOUS)
@enum(BoundType, NONE, LESS_THAN, GREATER_THAN, LESS_AND_GREATER_THAN, INTERVAL, EQUAL_TO)
@enum(ObjectiveType, SINGLE_VARIABLE, SCALAR_AFFINE, SCALAR_QUADRATIC)

mutable struct Optimizer <: MOI.AbstractOptimizer
    # The low-level Gurobi model.
    inner::Model
    # The Gurobi environment. If `nothing`, a new environment will be created
    # on `MOI.empty!`.
    env::Union{Nothing, Env}
    # The current user-provided parameters for the model.
    params::Dict{String, Any}

    # The next field is used to cleverly manage calls to `update_model!`.
    # `needs_update` is used to record whether an update should be called before
    # accessing a model attribute (such as the value of a RHS term).
    needs_update::Bool

    # A flag to keep track of MOI.Silent, which over-rides the OutputFlag
    # parameter.
    silent::Bool

    # An enum to remember what objective is currently stored in the model.
    objective_type::ObjectiveType

    # A flag to keep track of MOI.FEASIBILITY_SENSE, since Gurobi only stores
    # MIN_SENSE or MAX_SENSE. This allows us to differentiate between MIN_SENSE
    # and FEASIBILITY_SENSE.
    is_feasibility::Bool

    # A mapping from the MOI.VariableIndex to the Gurobi column. VariableInfo
    # also stores some additional fields like what bounds have been added, the
    # variable type, and the names of SingleVariable-in-Set constraints.
    last_variable_index::Int

    # An index that is incremented for each new constraint (regardless of type).
    # We can check if a constraint is valid by checking if it is in the correct
    # xxx_constraint_info. We should _not_ reset this to zero, since then new
    # constraints cannot be distinguished from previously created ones.

    # ScalarAffineFunction{Float64}-in-Set storage.
    last_affine_constraint_index::Int
    # ScalarQuadraticFunction{Float64}-in-Set storage.
    last_quadratic_constraint_index::Int
    # VectorOfVariables-in-Set storage.
    last_sos_constraint_index::Int

    # Note: we do not have a singlevariable_constraint_info dictionary. Instead,
    # data associated with these constraints are stored in the VariableInfo
    # objects.

    # These two flags allow us to distinguish between FEASIBLE_POINT and
    # INFEASIBILITY_CERTIFICATE when querying VariablePrimal and ConstraintDual.
    has_unbounded_ray::Bool
    has_infeasibility_cert::Bool

    # A helper cache for calling CallbackVariablePrimal.
    callback_variable_primal::Vector{Float64}

    """
        Optimizer(env = nothing; kwargs...)

    Create a new Optimizer object.

    You can share Gurobi `Env`s between models by passing an instance of `Env`
    as the first argument. By default, a new environment is created for every
    model.

    Note that we set the parameter `InfUnbdInfo` to `1` rather than the default
    of `0` so that we can query infeasibility certificates. Users are, however,
    free to over-ride this as follows `Optimizer(InfUndbInfo=0)`. In addition,
    we also set `QCPDual` to `1` to enable duals in QCPs. Users can override
    this by passing `Optimizer(QCPDual=0)`.
    """
    function Optimizer(env::Union{Nothing, Env} = nothing; kwargs...)
        model = new()
        model.env = env
        model.silent = false
        model.params = Dict{String, Any}()
        model.last_variable_index = 0
        model.last_affine_constraint_index = 0
        model.last_quadratic_constraint_index = 0
        model.last_sos_constraint_index = 0
        model.callback_variable_primal = Float64[]
        MOI.empty!(model)  # MOI.empty!(model) re-sets the `.inner` field.
        for (name, value) in kwargs
            model.params[string(name)] = value
            setparam!(model.inner, string(name), value)
        end
        if !haskey(model.params, "InfUnbdInfo")
            MOI.set(model, MOI.RawParameter("InfUnbdInfo"), 1)
        end
        if !haskey(model.params, "QCPDual")
            MOI.set(model, MOI.RawParameter("QCPDual"), 1)
        end
        return model
    end
end

Base.show(io::IO, model::Optimizer) = show(io, model.inner)

function MOI.empty!(model::Optimizer)
    if model.env === nothing
        model.inner = Model(Env(), "", finalize_env = true)
    else
        model.inner = Model(model.env, "", finalize_env = false)
    end
    for (name, value) in model.params
        setparam!(model.inner, name, value)
    end
    if model.silent
        # Set the parameter on the internal model, but don't modify the entry in
        # model.params so that if Silent() is set to `true`, the user-provided
        # value will be restored.
        setparam!(model.inner, "OutputFlag", 0)
    end
    model.needs_update = false
    model.objective_type = SCALAR_AFFINE
    model.is_feasibility = true
    model.last_variable_index = 0
    model.last_affine_constraint_index = 0
    model.last_quadratic_constraint_index = 0
    model.last_sos_constraint_index = 0
    model.has_unbounded_ray = false
    model.has_infeasibility_cert = false
    empty!(model.callback_variable_primal)
    return
end

function MOI.is_empty(model::Optimizer)
    model.needs_update && return false
    model.objective_type != SCALAR_AFFINE && return false
    model.is_feasibility == false && return false
    model.last_variable_index != 0 && return false
    model.last_affine_constraint_index != 0 && return false
    model.last_quadratic_constraint_index != 0 && return false
    model.last_sos_constraint_index != 0 && return false
    model.has_unbounded_ray && return false
    model.has_infeasibility_cert && return false
    length(model.callback_variable_primal) != 0 && return false
    return true
end

"""
    _require_update(model::Optimizer)

Sets the `model.needs_update` flag. Call this at the end of any mutating method.
"""
function _require_update(model::Optimizer)
    model.needs_update = true
    return
end

"""
    _require_update(model::Optimizer)

Calls `update_model!`, but only if the `model.needs_update` flag is set.
"""
function _update_if_necessary(model::Optimizer)
    if model.needs_update
        update_model!(model.inner)
        model.needs_update = false
    end
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Gurobi"

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{F}
) where {F <: Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64}
}}
    return true
end

function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.SingleVariable}, ::Type{F}
) where {F <: Union{
    MOI.EqualTo{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64},
    MOI.Interval{Float64}, MOI.ZeroOne, MOI.Integer,
    MOI.Semicontinuous{Float64}, MOI.Semiinteger{Float64}
}}
    return true
end

function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{F}
) where {F <: Union{MOI.SOS1{Float64}, MOI.SOS2{Float64}}}
    return true
end

# We choose _not_ to support ScalarAffineFunction-in-Interval and
# ScalarQuadraticFunction-in-Interval because Gurobi introduces some slack
# variables that makes it hard to keep track of the column indices.

function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{F}
) where {F <: Union{
    MOI.EqualTo{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64}
}}
    return true
end

function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{F}
) where {F <: Union{
    MOI.EqualTo{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64}
}}
    return true
end

const SCALAR_SETS = Union{
    MOI.GreaterThan{Float64}, MOI.LessThan{Float64},
    MOI.EqualTo{Float64}, MOI.Interval{Float64}
}

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = false
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{<:MOI.ConstraintIndex}) = false
MOI.supports(::Optimizer, ::MOI.ObjectiveFunctionType) = true

MOI.supports(::Optimizer, ::MOI.Name) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.ConstraintSet, c) = true
MOI.supports(::Optimizer, ::MOI.ConstraintFunction, c) = true
MOI.supports(::Optimizer, ::MOI.ConstraintPrimal, c) = true
MOI.supports(::Optimizer, ::MOI.ConstraintDual, c) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ListOfConstraintIndices) = false
MOI.supports(::Optimizer, ::MOI.RawStatusString) = true
MOI.supports(::Optimizer, ::MOI.RawParameter) = true

function MOI.set(model::Optimizer, param::MOI.RawParameter, value)
    model.params[param.name] = value
    setparam!(model.inner, param.name, value)
    return
end

function MOI.get(model::Optimizer, param::MOI.RawParameter)
    return getparam(model.inner, param.name)
end

MOI.Utilities.supports_default_copy_to(::Optimizer, ::Bool) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kwargs...)
    return MOI.Utilities.automatic_copy_to(dest, src; kwargs...)
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    return MOI.AbstractVariableAttribute[MOI.VariableName()]
end

function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    attributes = [
        MOI.ObjectiveSense(),
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    ]
    if MOI.get(model, MOI.Name()) != ""
        push!(attributes, MOI.Name())
    end
    return attributes
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintAttributesSet)
    return MOI.AbstractConstraintAttribute[MOI.ConstraintName()]
end

function _indices_and_coefficients(
    indices::AbstractVector{Int}, coefficients::AbstractVector{Float64},
    model::Optimizer, f::MOI.ScalarAffineFunction{Float64}
)
    for (i, term) in enumerate(f.terms)
        indices[i] = term.variable_index.value
        coefficients[i] = term.coefficient
    end
    return indices, coefficients
end

function _indices_and_coefficients(
    model::Optimizer, f::MOI.ScalarAffineFunction{Float64}
)
    f_canon = MOI.Utilities.canonical(f)
    nnz = length(f_canon.terms)
    indices = Vector{Int}(undef, nnz)
    coefficients = Vector{Float64}(undef, nnz)
    _indices_and_coefficients(indices, coefficients, model, f_canon)
    return indices, coefficients
end

function _indices_and_coefficients(
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector{Float64},
    indices::AbstractVector{Int}, coefficients::AbstractVector{Float64},
    model::Optimizer, f::MOI.ScalarQuadraticFunction
)
    for (i, term) in enumerate(f.quadratic_terms)
        I[i] = term.variable_index_1.value
        J[i] = term.variable_index_2.value
        V[i] = term.coefficient
        # Gurobi returns a list of terms. MOI requires 0.5 x' Q x. So, to get
        # from
        #   Gurobi -> MOI => multiply diagonals by 2.0
        #   MOI -> Gurobi => multiply diagonals by 0.5
        # Example: 2x^2 + x*y + y^2
        #   |x y| * |a b| * |x| = |ax+by bx+cy| * |x| = 0.5ax^2 + bxy + 0.5cy^2
        #           |b c|   |y|                   |y|
        #   Gurobi needs: (I, J, V) = ([0, 0, 1], [0, 1, 1], [2, 1, 1])
        #   MOI needs:
        #     [SQT(4.0, x, x), SQT(1.0, x, y), SQT(2.0, y, y)]
        if I[i] == J[i]
            V[i] *= 0.5
        end
    end
    for (i, term) in enumerate(f.affine_terms)
        indices[i] = term.variable_index.value
        coefficients[i] = term.coefficient
    end
    return
end

function _indices_and_coefficients(
    model::Optimizer, f::MOI.ScalarQuadraticFunction
)
    f_canon = MOI.Utilities.canonical(f)
    nnz_quadratic = length(f_canon.quadratic_terms)
    nnz_affine = length(f_canon.affine_terms)
    I = Vector{Int}(undef, nnz_quadratic)
    J = Vector{Int}(undef, nnz_quadratic)
    V = Vector{Float64}(undef, nnz_quadratic)
    indices = Vector{Int}(undef, nnz_affine)
    coefficients = Vector{Float64}(undef, nnz_affine)
    _indices_and_coefficients(I, J, V, indices, coefficients, model, f_canon)
    return indices, coefficients, I, J, V
end

_sense_and_rhs(s::MOI.LessThan{Float64}) = (Cchar('<'), s.upper)
_sense_and_rhs(s::MOI.GreaterThan{Float64}) = (Cchar('>'), s.lower)
_sense_and_rhs(s::MOI.EqualTo{Float64}) = (Cchar('='), s.value)

###
### Variables
###

# Short-cuts to return the VariableInfo associated with an index.
function MOI.add_variable(model::Optimizer)
    # Initialize `VariableInfo` with a dummy `VariableIndex` and a column,
    # because we need `add_item` to tell us what the `VariableIndex` is.
    add_cvar!(model.inner, 0.0)
    _require_update(model)
    return MOI.VariableIndex(model.last_variable_index += 1)
end

function MOI.add_variables(model::Optimizer, N::Int)
    add_cvars!(model.inner, zeros(N))
    _require_update(model)
    indices = MOI.VariableIndex.(1:N .+ model.last_variable_index)
    model.last_variable_index += N
    return indices
end

function MOI.is_valid(model::Optimizer, v::MOI.VariableIndex)
    return 1 <= v.value <= model.last_variable_index
end

# function MOI.delete(model::Optimizer, v::MOI.VariableIndex)
#     error("delete for variables not available")
# end

MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String) = nothing

MOI.get(model::Optimizer, ::MOI.VariableName, v::MOI.VariableIndex) = ""

MOI.set(model::Optimizer, ::MOI.VariableName, v::MOI.VariableIndex, name::String) = nothing

###
### Objectives
###

function MOI.set(
    model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense
)
    if sense == MOI.MIN_SENSE
        set_sense!(model.inner, :minimize)
        model.is_feasibility = false
    elseif sense == MOI.MAX_SENSE
        set_sense!(model.inner, :maximize)
        model.is_feasibility = false
    elseif sense == MOI.FEASIBILITY_SENSE
        set_sense!(model.inner, :minimize)
        model.is_feasibility = true
    else
        error("Invalid objective sense: $(sense)")
    end
    _require_update(model)
    return
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveSense)
    _update_if_necessary(model)
    sense = model_sense(model.inner)
    if model.is_feasibility
        return MOI.FEASIBILITY_SENSE
    elseif sense == :maximize
        return MOI.MAX_SENSE
    elseif sense == :minimize
        return MOI.MIN_SENSE
    end
    error("Invalid objective sense: $(sense)")
end

function MOI.set(
    model::Optimizer, ::MOI.ObjectiveFunction{F}, f::F
) where {F <: MOI.SingleVariable}
    MOI.set(
        model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        convert(MOI.ScalarAffineFunction{Float64}, f)
    )
    model.objective_type = SINGLE_VARIABLE
    return
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunction{MOI.SingleVariable})
    obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    return convert(MOI.SingleVariable, obj)
end

function MOI.set(
    model::Optimizer, ::MOI.ObjectiveFunction{F}, f::F
) where {F <: MOI.ScalarAffineFunction{Float64}}
    if model.objective_type == SCALAR_QUADRATIC
        # We need to zero out the existing quadratic objective.
        delq!(model.inner)
    end
    num_vars = model.last_variable_index
    obj = zeros(Float64, num_vars)
    for term in f.terms
        obj[term.variable_index.value] += term.coefficient
    end
    # This update is needed because we might have added some variables.
    _update_if_necessary(model)
    set_dblattrarray!(model.inner, "Obj", 1, num_vars, obj)
    set_dblattr!(model.inner, "ObjCon", f.constant)
    _require_update(model)
    model.objective_type = SCALAR_AFFINE
end

function MOI.get(
    model::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}
)
    if model.objective_type == SCALAR_QUADRATIC
        error("Unable to get objective function. Currently: $(model.objective_type).")
    end
    _update_if_necessary(model)
    dest = zeros(model.last_variable_index)
    get_dblattrarray!(dest, model.inner, "Obj", 1)
    terms = MOI.ScalarAffineTerm{Float64}[]
    for i in eachindex(dest)
        coefficient = dest[i]
        iszero(coefficient) && continue
        push!(terms, MOI.ScalarAffineTerm(coefficient, MOI.VariableIndex(i)))
    end
    constant = get_dblattr(model.inner, "ObjCon")
    return MOI.ScalarAffineFunction(terms, constant)
end

function MOI.set(
    model::Optimizer, ::MOI.ObjectiveFunction{F}, f::F
) where {F <: MOI.ScalarQuadraticFunction{Float64}}
    affine_indices, affine_coefficients, I, J, V = _indices_and_coefficients(model, f)
    _update_if_necessary(model)
    # We need to zero out any existing linear objective.
    obj = zeros(model.last_variable_index)
    for (i, c) in zip(affine_indices, affine_coefficients)
        obj[i] = c
    end
    set_dblattrarray!(model.inner, "Obj", 1, length(obj), obj)
    set_dblattr!(model.inner, "ObjCon", f.constant)
    # We need to zero out the existing quadratic objective.
    delq!(model.inner)
    add_qpterms!(model.inner, I, J, V)
    _require_update(model)
    model.objective_type = SCALAR_QUADRATIC
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}
)
    _update_if_necessary(model)
    dest = zeros(model.last_variable_index)
    get_dblattrarray!(dest, model.inner, "Obj", 1)
    terms = MOI.ScalarAffineTerm{Float64}[]
    for i in eachindex(dest)
        coefficient = dest[i]
        iszero(coefficient) && continue
        push!(terms, MOI.ScalarAffineTerm(coefficient, MOI.VariableIndex(i)))
    end
    constant = get_dblattr(model.inner, "ObjCon")
    q_terms = MOI.ScalarQuadraticTerm{Float64}[]
    I, J, V = getq(model.inner)
    for (i, j, v) in zip(I, J, V)
        iszero(v) && continue
        # See note in `_indices_and_coefficients`.
        new_v = i == j ? 2v : v
        push!(
            q_terms,
            MOI.ScalarQuadraticTerm(
                new_v,
                MOI.VariableIndex(i), # Maybe i + 1
                MOI.VariableIndex(j)  # Maybe j + 1
            )
        )
    end
    return MOI.ScalarQuadraticFunction(terms, q_terms, constant)
end

function MOI.modify(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.ScalarConstantChange{Float64}
)
    set_dblattr!(model.inner, "ObjCon", chg.new_constant)
    _require_update(model)
    return
end

##
##  SingleVariable-in-Set constraints.
##

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable}
)
    # TODO : Test for Set type!
    return 1 <= c.value <= model.last_variable_index
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable}
)
    MOI.throw_if_not_valid(model, c)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.set(
    model::Optimizer, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable}, ::MOI.SingleVariable
)
    return throw(MOI.SettingSingleVariableFunctionNotAllowed())
end

_bounds(s::MOI.GreaterThan{Float64}) = (s.lower, nothing)
_bounds(s::MOI.LessThan{Float64}) = (nothing, s.upper)
_bounds(s::MOI.EqualTo{Float64}) = (s.value, s.value)
_bounds(s::MOI.Interval{Float64}) = (s.lower, s.upper)

function MOI.add_constraint(
    model::Optimizer, f::MOI.SingleVariable, s::S
) where {S <: SCALAR_SETS}
    index = MOI.ConstraintIndex{MOI.SingleVariable, typeof(s)}(f.variable.value)
    MOI.set(model, MOI.ConstraintSet(), index, s)
    return index
end

function MOI.add_constraints(
    model::Optimizer, f::Vector{MOI.SingleVariable}, s::Vector{S}
) where {S <: SCALAR_SETS}
    indices = [
        MOI.ConstraintIndex{MOI.SingleVariable, eltype(s)}(fi.variable.value)
        for fi in f
    ]
    _set_bounds(model, indices, s)
    return indices
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    set_dblattrelement!(model.inner, "UB", c.value, Inf)
    _require_update(model)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    set_dblattrelement!(model.inner, "LB", c.value, -Inf)
    _require_update(model)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    column = c.value
    set_dblattrelement!(model.inner, "LB", column, -Inf)
    set_dblattrelement!(model.inner, "UB", column, Inf)
    _require_update(model)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    column = c.value
    set_dblattrelement!(model.inner, "LB", column, -Inf)
    set_dblattrelement!(model.inner, "UB", column, Inf)
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    _update_if_necessary(model)
    lower = get_dblattrelement(model.inner, "LB", c.value)
    return MOI.GreaterThan(lower)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    _update_if_necessary(model)
    upper = get_dblattrelement(model.inner, "UB", c.value)
    return MOI.LessThan(upper)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    _update_if_necessary(model)
    lower = get_dblattrelement(model.inner, "LB", c.value)
    return MOI.EqualTo(lower)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    _update_if_necessary(model)
    column = c.value
    lower = get_dblattrelement(model.inner, "LB", column)
    upper = get_dblattrelement(model.inner, "UB", column)
    return MOI.Interval(lower, upper)
end

function _set_bounds(
    model::Optimizer,
    indices::Vector{MOI.ConstraintIndex{MOI.SingleVariable, S}},
    sets::Vector{S}
) where {S}
    lower_columns, lower_values = Int[], Float64[]
    upper_columns, upper_values = Int[], Float64[]
    for (c, s) in zip(indices, sets)
        lower, upper = _bounds(s)
        column = c.value
        if lower !== nothing
            push!(lower_columns, column)
            push!(lower_values, lower)
        end
        if upper !== nothing
            push!(upper_columns, column)
            push!(upper_values, upper)
        end
    end
    if length(lower_columns) > 0
        set_dblattrlist!(model.inner, "LB", lower_columns, lower_values)
    end
    if length(upper_columns) > 0
        set_dblattrlist!(model.inner, "UB", upper_columns, upper_values)
    end
    _require_update(model)
end

function MOI.set(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, S}, s::S
) where {S<:SCALAR_SETS}
    MOI.throw_if_not_valid(model, c)
    lower, upper = _bounds(s)
    column = c.value
    if lower !== nothing
        set_dblattrelement!(model.inner, "LB", column, lower)
    end
    if upper !== nothing
        set_dblattrelement!(model.inner, "UB", column, upper)
    end
    _require_update(model)
    return
end

function MOI.add_constraint(
    model::Optimizer, f::MOI.SingleVariable, ::MOI.ZeroOne
)
    column = f.variable.value
    set_charattrelement!(model.inner, "VType", column, Char('B'))
    _require_update(model)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.ZeroOne}(column)
end

function MOI.delete(
    model::Optimizer, c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.ZeroOne}
)
    MOI.throw_if_not_valid(model, c)
    set_charattrelement!(model.inner, "VType", c.value, Char('C'))
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.ZeroOne}
)
    MOI.throw_if_not_valid(model, c)
    return MOI.ZeroOne()
end

function MOI.add_constraint(
    model::Optimizer, f::MOI.SingleVariable, ::MOI.Integer
)
    column = f.variable.value
    set_charattrelement!(model.inner, "VType", column, Char('I'))
    _require_update(model)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.Integer}(column)
end

function MOI.delete(
    model::Optimizer, c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Integer}
)
    MOI.throw_if_not_valid(model, c)
    set_charattrelement!(model.inner, "VType", c.value, Char('C'))
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Integer}
)
    MOI.throw_if_not_valid(model, c)
    return MOI.Integer()
end

function MOI.add_constraint(
    model::Optimizer, f::MOI.SingleVariable, s::MOI.Semicontinuous{Float64}
)
    column = f.variable.value
    # _throw_if_existing_lower(info.bound, info.type, typeof(s), f.variable)
    # _throw_if_existing_upper(info.bound, info.type, typeof(s), f.variable)
    set_charattrelement!(model.inner, "VType", column, Char('S'))
    set_dblattrelement!(model.inner, "LB", column, s.lower)
    set_dblattrelement!(model.inner, "UB", column, s.upper)
    _require_update(model)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.Semicontinuous{Float64}}(column)
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Semicontinuous{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    column = c.value
    set_charattrelement!(model.inner, "VType", column, Char('C'))
    set_dblattrelement!(model.inner, "LB", column, -Inf)
    set_dblattrelement!(model.inner, "UB", column, Inf)
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Semicontinuous{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    column = c.value
    _update_if_necessary(model)
    lower = get_dblattrelement(model.inner, "LB", column)
    upper = get_dblattrelement(model.inner, "UB", column)
    return MOI.Semicontinuous(lower, upper)
end

function MOI.add_constraint(
    model::Optimizer, f::MOI.SingleVariable, s::MOI.Semiinteger{Float64}
)
    column = f.variable.value
    # _throw_if_existing_lower(info.bound, info.type, typeof(s), f.variable)
    # _throw_if_existing_upper(info.bound, info.type, typeof(s), f.variable)
    set_charattrelement!(model.inner, "VType", column, Char('N'))
    set_dblattrelement!(model.inner, "LB", column, s.lower)
    set_dblattrelement!(model.inner, "UB", column, s.upper)
    _require_update(model)
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.Semiinteger{Float64}}(column)
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Semiinteger{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    column = c.value
    set_charattrelement!(model.inner, "VType", column, Char('C'))
    set_dblattrelement!(model.inner, "LB", column, -Inf)
    set_dblattrelement!(model.inner, "UB", column, Inf)
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Semiinteger{Float64}}
)
    MOI.throw_if_not_valid(model, c)
    column = c.value
    _update_if_necessary(model)
    lower = get_dblattrelement(model.inner, "LB", column)
    upper = get_dblattrelement(model.inner, "UB", column)
    return MOI.Semiinteger(lower, upper)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintName,
    c::MOI.ConstraintIndex
)
    MOI.throw_if_not_valid(model, c)
    return ""
end

function MOI.set(
    model::Optimizer, ::MOI.ConstraintName,
    c::MOI.ConstraintIndex, name::String
)
    MOI.throw_if_not_valid(model, c)
    return
end

###
### ScalarAffineFunction-in-Set
###

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}
) where {S}
    # TODO Test for type S?
    return 1 <= c.value <= model.last_affine_constraint_index
end

function MOI.add_constraint(
    model::Optimizer, f::MOI.ScalarAffineFunction{Float64},
    s::Union{MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}}
)
    if !iszero(f.constant)
        throw(MOI.ScalarFunctionConstantNotZero{Float64, typeof(f), typeof(s)}(f.constant))
    end
    model.last_affine_constraint_index += 1
    indices, coefficients = _indices_and_coefficients(model, f)
    sense, rhs = _sense_and_rhs(s)
    add_constr!(model.inner, indices, coefficients, sense, rhs)
    _require_update(model)
    return MOI.ConstraintIndex{typeof(f), typeof(s)}(model.last_affine_constraint_index)
end

function MOI.add_constraints(
    model::Optimizer, f::Vector{MOI.ScalarAffineFunction{Float64}},
    s::Vector{<:Union{MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}}}
)
    if length(f) != length(s)
        error("Number of functions does not equal number of sets.")
    end
    canonicalized_functions = MOI.Utilities.canonical.(f)

    # First pass: compute number of non-zeros to allocate space.
    nnz = 0
    for fi in canonicalized_functions
        if !iszero(fi.constant)
            throw(MOI.ScalarFunctionConstantNotZero{Float64, eltype(f), eltype(s)}(fi.constant))
        end
        nnz += length(fi.terms)
    end
    # Initialize storage
    indices = Vector{MOI.ConstraintIndex{eltype(f), eltype(s)}}(undef, length(f))
    row_starts = Vector{Int}(undef, length(f) + 1)
    row_starts[1] = 1
    columns = Vector{Int}(undef, nnz)
    coefficients = Vector{Float64}(undef, nnz)
    senses = Vector{Cchar}(undef, length(f))
    rhss = Vector{Float64}(undef, length(f))
    # Second pass: loop through, passing views to _indices_and_coefficients.
    for (i, (fi, si)) in enumerate(zip(canonicalized_functions, s))
        senses[i], rhss[i] = _sense_and_rhs(si)
        row_starts[i + 1] = row_starts[i] + length(fi.terms)
        _indices_and_coefficients(
            view(columns, row_starts[i]:row_starts[i + 1] - 1),
            view(coefficients, row_starts[i]:row_starts[i + 1] - 1),
            model, fi
        )
        model.last_affine_constraint_index += 1
        indices[i] = MOI.ConstraintIndex{eltype(f), eltype(s)}(model.last_affine_constraint_index)
    end
    pop!(row_starts)  # Gurobi doesn't need the final row start.
    add_constrs!(model.inner, row_starts, columns, coefficients, senses, rhss)
    _require_update(model)
    return indices
end

# function MOI.delete(
#     model::Optimizer,
#     c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}}
# )
#     error("Not implemented")
#     return
# end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}
) where {S}
    _update_if_necessary(model)
    rhs = get_dblattrelement(model.inner, "RHS", c.value)
    return S(rhs)
end

function MOI.set(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}, s::S
) where {S}
    set_dblattrelement!(model.inner, "RHS", c.value, MOI.constant(s))
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}
) where {S}
    _update_if_necessary(model)
    sparse_a = SparseArrays.sparse(get_constrs(model.inner, c.value, 1)')
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col, val) in zip(sparse_a.rowval, sparse_a.nzval)
        iszero(val) && continue
        push!(
            terms,
            MOI.ScalarAffineTerm(
                val,
                MOI.VariableIndex(col)
            )
        )
    end
    return MOI.ScalarAffineFunction(terms, 0.0)
end

MOI.get(model::Optimizer, ::Type{<:MOI.ConstraintIndex}, name::String) = nothing

###
### ScalarQuadraticFunction-in-SCALAR_SET
###

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarQuadraticFunction{Float64}, s::SCALAR_SETS
)
    if !iszero(f.constant)
        throw(MOI.ScalarFunctionConstantNotZero{Float64, typeof(f), typeof(s)}(f.constant))
    end
    indices, coefficients, I, J, V = _indices_and_coefficients(model, f)
    sense, rhs = _sense_and_rhs(s)
    add_qconstr!(model.inner, indices, coefficients, I, J, V, sense, rhs)
    _require_update(model)
    model.last_quadratic_constraint_index += 1
    return MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, typeof(s)}(model.last_quadratic_constraint_index)
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, S}
) where {S}
    # TODO Test for S type?
    1 <= c.value <= model.last_quadratic_constraint_index
end

# function MOI.delete(
#     model::Optimizer,
#     c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, S}
# ) where {S}
#     error("Not implemented")
#     return
# end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, S}
) where {S}
    _update_if_necessary(model)
    rhs = get_dblattrelement(model.inner, "QCRHS", c.value)
    return S(rhs)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, S}
) where {S}
    _update_if_necessary(model)
    affine_cols, affine_coefficients, I, J, V = getqconstr(model.inner, c.value)
    affine_terms = MOI.ScalarAffineTerm{Float64}[]
    for (col, coef) in zip(affine_cols, affine_coefficients)
        iszero(coef) && continue
        push!(
            affine_terms,
            MOI.ScalarAffineTerm(coef, MOI.VariableIndex(col))
        )
    end
    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for (i, j, coef) in zip(I, J, V)
        new_coef = i == j ? 2coef : coef
        push!(
            quadratic_terms,
            MOI.ScalarQuadraticTerm(
                new_coef,
                MOI.VariableIndex(i),
                MOI.VariableIndex(j)
            )
        )
    end
    constant = get_dblattr(model.inner, "ObjCon")
    return MOI.ScalarQuadraticFunction(affine_terms, quadratic_terms, constant)
end

###
### VectorOfVariables-in-SOS{I|II}
###

const SOS = Union{MOI.SOS1{Float64}, MOI.SOS2{Float64}}

_sos_type(::MOI.SOS1) = :SOS1
_sos_type(::MOI.SOS2) = :SOS2

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.VectorOfVariables, S}
) where {S}
    # TODO Test for S type?
    1 <= c.value <= model.last_sos_constraint_index
end

function MOI.add_constraint(
    model::Optimizer, f::MOI.VectorOfVariables, s::SOS
)
    columns = getfield.(f.variables, :value)
    add_sos!(model.inner, _sos_type(s), columns, s.weights)
    model.last_sos_constraint_index += 1
    index = MOI.ConstraintIndex{MOI.VectorOfVariables, typeof(s)}(model.last_sos_constraint_index)
    _require_update(model)
    return index
end

# function MOI.delete(
#     model::Optimizer, c::MOI.ConstraintIndex{MOI.VectorOfVariables, <:SOS}
# )
#     error("Not implemented")
#     return
# end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.VectorOfVariables, S}
) where {S <: SOS}
    _update_if_necessary(model)
    sparse_a, _ = get_sos(model.inner, c.value, 1)
    return S(sparse_a.nzval)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.VectorOfVariables, S}
) where {S <: SOS}
    _update_if_necessary(model)
    sparse_a, _ = get_sos(model.inner, c.value, 1)
    indices = SparseArrays.nonzeroinds(sparse_a[1, :])
    return MOI.VectorOfVariables(
        MOI.VariableIndex.(indices)
    )
end

###
### Optimize methods.
###

function MOI.optimize!(model::Optimizer)
    # Note: although Gurobi will call update regardless, we do it now so that
    # the appropriate `needs_update` flag is set.
    _update_if_necessary(model)
    optimize(model.inner)
    model.has_infeasibility_cert =
    MOI.get(model, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
    model.has_unbounded_ray =
        MOI.get(model, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
    return
end

# These strings are taken directly from the following page of the online Gurobi
# documentation: https://www.com/documentation/8.1/refman/optimization_status_codes.html#sec:StatusCodes
const RAW_STATUS_STRINGS = [
    (MOI.OPTIMIZE_NOT_CALLED, "Model is loaded, but no solution information is available."),
    (MOI.OPTIMAL, "Model was solved to optimality (subject to tolerances), and an optimal solution is available."),
    (MOI.INFEASIBLE, "Model was proven to be infeasible."),
    (MOI.INFEASIBLE_OR_UNBOUNDED, "Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize."),
    (MOI.DUAL_INFEASIBLE, "Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize."),
    (MOI.OBJECTIVE_LIMIT, "Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available."),
    (MOI.ITERATION_LIMIT, "Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter."),
    (MOI.NODE_LIMIT, "Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter."),
    (MOI.TIME_LIMIT, "Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter."),
    (MOI.SOLUTION_LIMIT, "Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter."),
    (MOI.INTERRUPTED, "Optimization was terminated by the user."),
    (MOI.NUMERICAL_ERROR, "Optimization was terminated due to unrecoverable numerical difficulties."),
    (MOI.OTHER_LIMIT, "Unable to satisfy optimality tolerances; a sub-optimal solution is available."),
    (MOI.OTHER_ERROR, "An asynchronous optimization call was made, but the associated optimization run is not yet complete."),
    (MOI.OBJECTIVE_LIMIT, "User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached.")
]

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    status_code = get_status_code(model.inner)
    if 1 <= status_code <= length(RAW_STATUS_STRINGS)
        return RAW_STATUS_STRINGS[status_code][2]
    end
    return MOI.OTHER_ERROR
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    status_code = get_status_code(model.inner)
    if 1 <= status_code <= length(RAW_STATUS_STRINGS)
        return RAW_STATUS_STRINGS[status_code][1]
    end
    return MOI.OTHER_ERROR
end

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    stat = get_status(model.inner)
    if stat == :optimal
        return MOI.FEASIBLE_POINT
    elseif stat == :solution_limit
        return MOI.FEASIBLE_POINT
    elseif (stat == :inf_or_unbd || stat == :unbounded) && _has_primal_ray(model)
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif stat == :suboptimal
        return MOI.FEASIBLE_POINT
    elseif is_mip(model.inner) && get_sol_count(model.inner) > 0
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

function _has_dual_ray(model::Optimizer)
    try
        # Note: for performance reasons, we try to get 1 element because for
        # some versions of Gurobi, we cannot query 0 elements without error.
        get_dblattrarray(model.inner, "FarkasDual", 1, 1)
        return true
    catch ex
        if isa(ex, GurobiError)
            return false
        else
            rethrow(ex)
        end
    end
end

function MOI.get(model::Optimizer, ::MOI.DualStatus)
    stat = get_status(model.inner)
    if is_mip(model.inner)
        return MOI.NO_SOLUTION
    elseif is_qcp(model.inner) && MOI.get(model, MOI.RawParameter("QCPDual")) != 1
        return MOI.NO_SOLUTION
    elseif stat == :optimal
        return MOI.FEASIBLE_POINT
    elseif stat == :solution_limit
        return MOI.FEASIBLE_POINT
    elseif (stat == :inf_or_unbd || stat == :infeasible) && _has_dual_ray(model)
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif stat == :suboptimal
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

function _has_primal_ray(model::Optimizer)
    try
        # Note: for performance reasons, we try to get 1 element because for
        # some versions of Gurobi, we cannot query 0 elements without error.
        get_dblattrarray(model.inner, "UnbdRay", 1, 1)
        return true
    catch ex
        if isa(ex, GurobiError)
            return false
        else
            rethrow(ex)
        end
    end
end

function MOI.get(model::Optimizer, ::MOI.VariablePrimal, x::MOI.VariableIndex)
    if model.has_unbounded_ray
        return get_dblattrelement(model.inner, "UnbdRay", x.value)
    else
        return get_dblattrelement(model.inner, "X", x.value)
    end
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.SingleVariable, <:Any}
)
    return MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, <:Any}
)
    row = c.value
    _update_if_necessary(model)
    rhs = get_dblattrelement(model.inner, "RHS", row)
    slack = get_dblattrelement(model.inner, "Slack", row)
    return rhs - slack
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, <:Any}
)
    row = c.value
    _update_if_necessary(model)
    rhs = get_dblattrelement(model.inner, "QCRHS", row)
    slack = get_dblattrelement(model.inner, "QCSlack", row)
    return rhs - slack
end

function _dual_multiplier(model::Optimizer)
    return MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE ? 1.0 : -1.0
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}
)
    _update_if_necessary(model)
    column = c.value
    x = get_dblattrelement(model.inner, "X", column)
    ub = get_dblattrelement(model.inner, "UB", column)
    if x ≈ ub
        return _dual_multiplier(model) * get_dblattrelement(model.inner, "RC", column)
    else
        return 0.0
    end
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}
)
    _update_if_necessary(model)
    column = c.value
    x = get_dblattrelement(model.inner, "X", column)
    lb = get_dblattrelement(model.inner, "LB", column)
    if x ≈ lb
        return _dual_multiplier(model) * get_dblattrelement(model.inner, "RC", column)
    else
        return 0.0
    end
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}
)
    return _dual_multiplier(model) * get_dblattrelement(model.inner, "RC", c.value)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{Float64}}
)
    return _dual_multiplier(model) * get_dblattrelement(model.inner, "RC", c.value)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}}
)
    if model.has_infeasibility_cert
        return -_dual_multiplier(model) * get_dblattrelement(model.inner, "FarkasDual", c.value)
    end
    return _dual_multiplier(model) * get_dblattrelement(model.inner, "Pi", c.value)
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}}
)
    return _dual_multiplier(model) * get_dblattrelement(model.inner, "QCPi", c.value)
end

MOI.get(model::Optimizer, ::MOI.ObjectiveValue) = get_dblattr(model.inner, "ObjVal")
MOI.get(model::Optimizer, ::MOI.ObjectiveBound) = get_dblattr(model.inner, "ObjBound")
MOI.get(model::Optimizer, ::MOI.SolveTime) = get_dblattr(model.inner, "RunTime")
MOI.get(model::Optimizer, ::MOI.SimplexIterations) = get_intattr(model.inner, "IterCount")
MOI.get(model::Optimizer, ::MOI.BarrierIterations) = get_intattr(model.inner, "BarIterCount")
MOI.get(model::Optimizer, ::MOI.NodeCount) = get_intattr(model.inner, "NodeCount")
MOI.get(model::Optimizer, ::MOI.RelativeGap) = get_dblattr(model.inner, "MIPGap")

MOI.supports(model::Optimizer, ::MOI.DualObjectiveValue) = true
MOI.get(model::Optimizer, ::MOI.DualObjectiveValue) = get_dblattr(model.inner, "ObjBound")

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.has_infeasibility_cert || model.has_unbounded_ray
        return 1
    end
    return get_intattr(model.inner, "SolCount")
end

function MOI.get(model::Optimizer, ::MOI.Silent)
    return model.silent
end

function MOI.set(model::Optimizer, ::MOI.Silent, flag::Bool)
    model.silent = flag
    output_flag = flag ? 0 : get(model.params, "OutputFlag", 1)
    setparam!(model.inner, "OutputFlag", output_flag)
    return
end

function MOI.get(model::Optimizer, ::MOI.Name)
    _update_if_necessary(model)
    return get_strattr(model.inner, "ModelName")
end

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    set_strattr!(model.inner, "ModelName", name)
    _require_update(model)
    return
end

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = model.last_variable_index
function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return MOI.VariableIndex.(1:model.last_variable_index)
end

MOI.get(model::Optimizer, ::MOI.RawSolver) = model.inner

function MOI.set(
    model::Optimizer, ::MOI.VariablePrimalStart, x::MOI.VariableIndex,
    value::Union{Nothing, Float64}
)
    if value !== nothing
        set_dblattrelement!(model.inner, "Start", x.value, value)
        _require_update(model)
    end
    return
end

function MOI.get(
    model::Optimizer, ::MOI.VariablePrimalStart, x::MOI.VariableIndex
)
    _update_if_necessary(model)
    get_dblattrelement!(model.inner, "Start", x.value)
end

MOI.supports(::Optimizer, ::MOI.ConstraintPrimalStart) = false
MOI.supports(::Optimizer, ::MOI.ConstraintDualStart) = false

# function MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{F, S}) where {F, S}
#     # TODO: this could be more efficient.
#     return length(MOI.get(model, MOI.ListOfConstraintIndices{F, S}()))
# end

_bound_enums(::Type{<:MOI.LessThan}) = (LESS_THAN, LESS_AND_GREATER_THAN)
_bound_enums(::Type{<:MOI.GreaterThan}) = (GREATER_THAN, LESS_AND_GREATER_THAN)
_bound_enums(::Type{<:MOI.Interval}) = (INTERVAL,)
_bound_enums(::Type{<:MOI.EqualTo}) = (EQUAL_TO,)
_bound_enums(::Any) = (nothing,)

_type_enums(::Type{MOI.ZeroOne}) = (BINARY,)
_type_enums(::Type{MOI.Integer}) = (INTEGER,)
_type_enums(::Type{<:MOI.Semicontinuous}) = (SEMICONTINUOUS,)
_type_enums(::Type{<:MOI.Semiinteger}) = (SEMIINTEGER,)
_type_enums(::Any) = (nothing,)

# function MOI.get(
#     model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, S}
# ) where {S}
#     indices = MOI.ConstraintIndex{MOI.SingleVariable, S}[]
#     for (key, info) in model.variable_info
#         if info.bound in _bound_enums(S) || info.type in _type_enums(S)
#             push!(indices, MOI.ConstraintIndex{MOI.SingleVariable, S}(key.value))
#         end
#     end
#     return sort!(indices, by = x -> x.value)
# end

# function MOI.get(
#     model::Optimizer,
#     ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, S}
# ) where {S}
#     error("Not implemented")
#     indices = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}[]
#     for (key, info) in model.affine_constraint_info
#         if typeof(info.set) == S
#             push!(indices, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(key))
#         end
#     end
#     return sort!(indices, by = x -> x.value)
# end

# function MOI.get(
#     model::Optimizer,
#     ::MOI.ListOfConstraintIndices{MOI.ScalarQuadraticFunction{Float64}, S}
# ) where {S}
#     error("Not implemented")
#     indices = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, S}[]
#     for (key, info) in model.quadratic_constraint_info
#         if typeof(info.set) == S
#             push!(indices, MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}, S}(key))
#         end
#     end
#     return sort!(indices, by = x -> x.value)
# end

# function MOI.get(
#     model::Optimizer, ::MOI.ListOfConstraintIndices{MOI.VectorOfVariables, S}
# ) where {S}
#     error("Not implemented")
#     indices = MOI.ConstraintIndex{MOI.VectorOfVariables, S}[]
#     for (key, info) in model.sos_constraint_info
#         if typeof(info.set) == S
#             push!(indices, MOI.ConstraintIndex{MOI.VectorOfVariables, S}(key))
#         end
#     end
#     return sort!(indices, by = x -> x.value)
# end

# function MOI.get(model::Optimizer, ::MOI.ListOfConstraints)
#     error("Not implemented")
#     constraints = Set{Any}()
#     for info in values(model.variable_info)
#         if info.bound == NONE
#         elseif info.bound == LESS_THAN
#             push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
#         elseif info.bound == GREATER_THAN
#             push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
#         elseif info.bound == LESS_AND_GREATER_THAN
#             push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
#             push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
#         elseif info.bound == EQUAL_TO
#             push!(constraints, (MOI.SingleVariable, MOI.EqualTo{Float64}))
#         elseif info.bound == INTERVAL
#             push!(constraints, (MOI.SingleVariable, MOI.Interval{Float64}))
#         end
#         if info.type == CONTINUOUS
#         elseif info.type == BINARY
#             push!(constraints, (MOI.SingleVariable, MOI.ZeroOne))
#         elseif info.type == INTEGER
#             push!(constraints, (MOI.SingleVariable, MOI.Integer))
#         elseif info.type == SEMICONTINUOUS
#             push!(constraints, (MOI.SingleVariable, MOI.Semicontinuous{Float64}))
#         elseif info.type == SEMIINTEGER
#             push!(constraints, (MOI.SingleVariable, MOI.Semiinteger{Float64}))
#         end
#     end
#     for info in values(model.affine_constraint_info)
#         push!(constraints, (MOI.ScalarAffineFunction{Float64}, typeof(info.set)))
#     end
#     for info in values(model.sos_constraint_info)
#         push!(constraints, (MOI.VectorOfVariables, typeof(info.set)))
#     end
#     return collect(constraints)
# end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType)
    if model.objective_type == SINGLE_VARIABLE
        return MOI.SINGLE_VARIABLE
    elseif model.objective_type == SCALAR_AFFINE
        return MOI.ScalarAffineFunction{Float64}
    else
        @assert model.objective_type == SCALAR_QUADRATIC
        return MOI.ScalarQuadraticFunction{Float64}
    end
end

function MOI.modify(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.ScalarCoefficientChange{Float64}
)
    chg_coeffs!(
        model.inner, c.value, chg.variable.value,
        chg.new_coefficient
    )
    _require_update(model)
end

function MOI.modify(
    model::Optimizer,
    c::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.ScalarCoefficientChange{Float64}
)
    set_dblattrelement!(
        model.inner, "Obj", chg.variable.value,
        chg.new_coefficient
    )
    _require_update(model)
end

"""
    _replace_with_matching_sparsity!(
        model::Optimizer,
        previous::MOI.ScalarAffineFunction,
        replacement::MOI.ScalarAffineFunction, row::Int
    )

Internal function, not intended for external use.

Change the linear constraint function at index `row` in `model` from
`previous` to `replacement`. This function assumes that `previous` and
`replacement` have exactly the same sparsity pattern w.r.t. which variables
they include and that both constraint functions are in canonical form (as
returned by `MOIU.canonical()`. Neither assumption is checked within the body
of this function.
"""
function _replace_with_matching_sparsity!(
    model::Optimizer,
    previous::MOI.ScalarAffineFunction,
    replacement::MOI.ScalarAffineFunction, row::Int
)
    rows = fill(Cint(row), length(replacement.terms))
    cols = [Cint(t.variable_index.value) for t in replacement.terms]
    coefs = MOI.coefficient.(replacement.terms)
    chg_coeffs!(model.inner, rows, cols, coefs)
    return
end

"""
    _replace_with_different_sparsity!(
        model::Optimizer,
        previous::MOI.ScalarAffineFunction,
        replacement::MOI.ScalarAffineFunction, row::Int
    )

Internal function, not intended for external use.

    Change the linear constraint function at index `row` in `model` from
`previous` to `replacement`. This function assumes that `previous` and
`replacement` may have different sparsity patterns.

This function (and `_replace_with_matching_sparsity!` above) are necessary
because in order to fully replace a linear constraint, we have to zero out the
current matrix coefficients and then set the new matrix coefficients. When the
sparsity patterns match, the zeroing-out step can be skipped.
"""
function _replace_with_different_sparsity!(
    model::Optimizer,
    previous::MOI.ScalarAffineFunction,
    replacement::MOI.ScalarAffineFunction, row::Int
)
    # First, zero out the old constraint function terms.
    rows = fill(Cint(row), length(previous.terms))
    cols = [Cint(t.variable_index.value) for t in previous.terms]
    coefs = fill(0.0, length(previous.terms))
    chg_coeffs!(model.inner, rows, cols, coefs)
    # Next, set the new constraint function terms.
    rows = fill(Cint(row), length(replacement.terms))
    cols = [Cint(t.variable_index.value) for t in replacement.terms]
    coefs = MOI.coefficient.(replacement.terms)
    chg_coeffs!(model.inner, rows, cols, coefs)
    return
end

"""
    _matching_sparsity_pattern(
        f1::MOI.ScalarAffineFunction{Float64},
        f2::MOI.ScalarAffineFunction{Float64}
    )

Internal function, not intended for external use.

Determines whether functions `f1` and `f2` have the same sparsity pattern
w.r.t. their constraint columns. Assumes both functions are already in
canonical form.
"""
function _matching_sparsity_pattern(
    f1::MOI.ScalarAffineFunction{Float64}, f2::MOI.ScalarAffineFunction{Float64}
)
    if axes(f1.terms) != axes(f2.terms)
        return false
    end
    for (f1_term, f2_term) in zip(f1.terms, f2.terms)
        if MOI.term_indices(f1_term) != MOI.term_indices(f2_term)
            return false
        end
    end
    return true
end

function MOI.set(
    model::Optimizer, ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, <:SCALAR_SETS},
    f::MOI.ScalarAffineFunction{Float64}
)
    previous = MOI.get(model, MOI.ConstraintFunction(), c)
    MOI.Utilities.canonicalize!(previous)
    replacement = MOI.Utilities.canonical(f)
    _update_if_necessary(model)
    # If the previous and replacement constraint functions have exactly
    # the same sparsity pattern, then we can take a faster path by just
    # passing the replacement terms to the model. But if their sparsity
    # patterns differ, then we need to first zero out the previous terms
    # and then set the replacement terms.
    row = c.value
    if _matching_sparsity_pattern(previous, replacement)
        _replace_with_matching_sparsity!(model, previous, replacement, row)
    else
        _replace_with_different_sparsity!(model, previous, replacement, row)
    end
    current_rhs = get_dblattrelement(model.inner, "RHS", row)
    new_rhs = current_rhs - (replacement.constant - previous.constant)
    set_dblattrelement!(model.inner, "RHS", row, new_rhs)
    _require_update(model)
    return
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintBasisStatus,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}
) where {S <: SCALAR_SETS}
    row = c.value
    _update_if_necessary(model)
    cbasis = get_intattrelement(model.inner, "CBasis", row)
    if cbasis == 0
        return MOI.BASIC
    elseif cbasis == -1
        return MOI.NONBASIC
    else
        error("CBasis value of $(cbasis) isn't defined.")
    end
end

function MOI.get(
    model::Optimizer, ::MOI.ConstraintBasisStatus,
    c::MOI.ConstraintIndex{MOI.SingleVariable, S}
) where {S <: SCALAR_SETS}
    column = c.value
    _update_if_necessary(model)
    vbasis = get_intattrelement(model.inner, "VBasis", column)
    if vbasis == 0
        return MOI.BASIC
    elseif vbasis == -1
        if S <: MOI.LessThan
            return MOI.BASIC
        elseif !(S <: MOI.Interval)
            return MOI.NONBASIC
        else
            return MOI.NONBASIC_AT_LOWER
        end
    elseif vbasis == -2
        MOI.NONBASIC_AT_UPPER
        if S <: MOI.GreaterThan
            return MOI.BASIC
        elseif !(S <: MOI.Interval)
            return MOI.NONBASIC
        else
            return MOI.NONBASIC_AT_UPPER
        end
    elseif vbasis == -3
        return MOI.SUPER_BASIC
    else
        error("VBasis value of $(vbasis) isn't defined.")
    end
end

# ==============================================================================
#    Callbacks in Gurobi
# ==============================================================================

struct CallbackFunction <: MOI.AbstractOptimizerAttribute end

function MOI.set(model::Optimizer, ::CallbackFunction, f::Function)
    set_callback_func!(model.inner, f)
    update_model!(model.inner)
    return
end

struct CallbackVariablePrimal <: MOI.AbstractVariableAttribute end

function load_callback_variable_primal(model, cb_data, cb_where)
    if cb_where != CB_MIPSOL
        error("`load_callback_variable_primal` must be called from `CB_MIPSOL`.")
    end
    resize!(model.callback_variable_primal, model.last_variable_index)
    cbget_mipsol_sol(cb_data, cb_where, model.callback_variable_primal)
    return
end

# Note: you must call load_callback_variable_primal first.
function MOI.get(
    model::Optimizer, ::CallbackVariablePrimal, x::MOI.VariableIndex
)
    return model.callback_variable_primal[x.value]
end

"""
    function cblazy!(
        cb_data::CallbackData, model::Optimizer,
        f::MOI.ScalarAffineFunction{Float64},
        s::Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}
    )

Add a lazy cut to the model `m`.

You must have the option `LazyConstraints` set  via `Optimizer(LazyConstraint=1)`.
This can only be called in a callback from `CB_MIPSOL`.
"""
function cblazy!(
    cb_data::CallbackData, model::Optimizer,
    f::MOI.ScalarAffineFunction{Float64},
    s::Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}
)
    indices, coefficients = _indices_and_coefficients(model, f)
    sense, rhs = _sense_and_rhs(s)
    return cblazy(cb_data, Cint.(indices), coefficients, Char(sense), rhs)
end
