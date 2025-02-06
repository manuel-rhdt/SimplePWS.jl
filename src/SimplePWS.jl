module SimplePWS

include("models.jl")
include("armodel.jl")
include("mutual_information.jl")

import .Models: BirthDeathModel, NonlinearModel
import .MutualInformation: mutual_information
import .armodel: ARModel, generate_stable_ar_coefficients

export BirthDeathModel, NonlinearModel, ARModel, mutual_information

end
