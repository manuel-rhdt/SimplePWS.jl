module SimplePWS

include("models.jl")
include("mutual_information.jl")

import .Models: BirthDeathModel, NonlinearModel
import .MutualInformation: mutual_information

export BirthDeathModel, NonlinearModel, mutual_information

end
