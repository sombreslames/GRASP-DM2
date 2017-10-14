#Ferrari Leon
#M1 ORO
#Version de test
#Julia JuMP
#DM1 - Metaheuristiques
using JuMP, GLPKMathProgInterface
include("myHeuristics.jl")
type Problem
   NBvariables::Int
   NBconstraints::Int
   Variables::Vector{Int}
   LeftMembers_Constraints::SparseMatrixCSC{Float64,Int64}
   RightMembers_Constraints::Vector{Int}
end
type CurrentSolution
   NBconstraints::Int
   NBvariables::Int
   CurrentObjectiveValue::Int
   Variables::Vector{Int}
   CurrentVariables::Vector{Int}
   LeftMembers_Constraints::SparseMatrixCSC{Float64,Int64}
   LastRightMemberValue_Constraint::Vector{Int}
   Utility::Array{Float64,2}
   Freedom::Vector{Int}
end
type Result
   Name::String
   NBvariables::Int
   GLPKObj::Int
   HeurObj::Int
   GLPKTime
   HeurTime
   Result() = new()#Allow to create unintialized type
end
type ProbStat
   Max::Vector{Float64}
   Average::Vector{Float64}
   Min::Vector{Float64}
   NBdone::Vector{Float64}
end
function ReadFile(FileName::String)
   workingfile    = open(FileName)
   NBcons,NBvar   = parse.(split(readline(workingfile)))
   Coef           = parse.(split(readline(workingfile)))
   LeftMembers_Constraints    = spzeros(NBcons,NBvar)
   RightMembers_Constraints   = Vector(NBcons)
   for i = 1:1:NBcons
         readline(workingfile)
         RightMembers_Constraints[i]=1
         for val in split(readline(workingfile))
            LeftMembers_Constraints[i, parse(val)]=1
         end
   end
   close(workingfile)
   return Problem(NBvar, NBcons, Coef, LeftMembers_Constraints, RightMembers_Constraints)
end

#GETTING DATA FRON FILE
nbProb = 1
FileList = readdir("./Data")
dir = pwd()
Resume              = Vector{Result}(20)

for i in eachindex(FileList)
   #MODEL CONSTRUCTION
   m           = Model(solver=GLPKSolverMIP())
   #READING DATA FROM FILE
   BPP = ReadFile(string("./Data/",FileList[i]))
   if nbProb <= 1 && BPP.NBvariables == 100 && BPP.NBconstraints == 500
      println("On effectue 100 fois le probleme : ",FileList[i])
      #ProbTemp = Result();

      LambdaVal   = [0.5, 0.6, 0.75, 0.9]
      LambdaProba = [0.25,0.25,0.25,0.25]

      println("Before the 100 run we got these Lambda :")
      println(LambdaVal)
      println("With these probabilities :")
      println(LambdaProba)
      Stat        = ProbStat(zeros(Float64,4),zeros(Float64,4),zeros(Float64,4),zeros(Float64,4))
      fill!(Stat.Min, typemax(Float64))
      for j in 1:1:15

         for k in 1:1:100
            indLa,Lambda   = ReactiveGrasp(LambdaProba,LambdaVal)
            CS             = GraspConstruction(CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, BPP.Variables,zeros(BPP.NBvariables), BPP.LeftMembers_Constraints, zeros(BPP.NBconstraints), zeros(2,BPP.NBvariables), zeros(BPP.NBvariables)),Lambda)
            #Effectuer la recherche locale ici
            CS             = LocalSearch(CS,convert(Int,Lambda*CS.NBvariables))
            if CS.CurrentObjectiveValue > Stat.Max[indLa]
               Stat.Max[indLa] = CS.CurrentObjectiveValue
            elseif CS.CurrentObjectiveValue < Stat.Min[indLa]
               Stat.Min[indLa] = CS.CurrentObjectiveValue
            end
            Stat.NBdone[indLa]  += 1
            Stat.Average[indLa] += CS.CurrentObjectiveValue
         end
      end
      for d in 1:1:4
         Stat.Average[d] = Stat.Average[d]/Stat.NBdone[d]
      end
      LambdaProba       = UpdateReactiveGrasp(LambdaProba, Stat.Average,Stat.Min,Stat.Max)
      println("Maximum found : ",Stat.Max)
      println("After the 1500 run we got these probabilities :")
      println(LambdaProba)
      println("Average objective value :",Stat.Average,Stat.NBdone)
      #=
      @variable( m,  x[1:BPP.NBvariables], Bin)
      @objective( m , Max, sum( BPP.Variables[j] * x[j] for j=1:BPP.NBvariables ) )
      @constraint( m , cte[i=1:BPP.NBconstraints], sum(BPP.LeftMembers_Constraints[i,j] * x[j] for j=1:BPP.NBvariables) <= BPP.RightMembers_Constraints[i] )
      #SOLVE IT AND DISPLAY THE RESULTS
      #--------------------------------
      ProbTemp.GLPKTime       = @elapsed solve(m) # solves the model
      ProbTemp.NBvariables    = BPP.NBvariables
      ProbTemp.Name           = string("./Data/",FileList[i])
      ProbTemp.GLPKObj        = getobjectivevalue(m)
      ProbTemp.HeurObj        = cs.CurrentObjectiveValue
      Resume[nbProb]          = deepcopy(ProbTemp)=#
      nbProb+=1
   elseif nbProb > 1
      break;
   end

end
#=
for resu in eachindex(Resume)
   println("For the problem ",Resume[resu].Name," with ",Resume[resu].NBvariables, " variables.")
   println("GLPK Solver Objective value : ",Resume[resu].GLPKObj," Time : ",Resume[resu].GLPKTime ) # getObjectiveValue(model_name) gives the optimum objective value
   println("Heuristics :",Resume[resu].HeurObj," Time : ",Resume[resu].HeurTime,"\n")
end=#
