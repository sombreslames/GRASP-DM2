#Ferrari Leon
#M1 ORO
#Version de test
#Julia JuMP
#DM1 - Metaheuristiques
using JuMP, GLPKMathProgInterface,PyPlot

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
   CurrentVarUsed::Vector{Int}
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
   Max::Float64
   Average::Vector{Float64}
   Min::Float64
   NBdone::Vector{Float64}
   Sum::Vector{Int64}
   BestSolution::CurrentSolution
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
   #if nbProb <= 4 && BPP.NBvariables <= 100 && BPP.NBconstraints <= 400
   # FileList[i] == "pb_100rnd0700.dat" ||
   if FileList[i] == "pb_100rnd0700.dat"  || FileList[i] == "pb_1000rnd0100.dat" || FileList[i] =="pb_100rnd0100.dat" || FileList[i] == "pb_2000rnd0100.dat"  || FileList[i] == "pb_200rnd0100.dat" || FileList[i] == "pb_500rnd0100.dat"
      println("Probleme : ",FileList[i])
      #ProbTemp = Result();

      AlphaVal   = [0.5, 0.6, 0.75, 0.9]
      AlphaProba = [0.25,0.25,0.25,0.25]

      println("Before the run we got these Lambda :")
      println(AlphaVal)
      println("With these probabilities :")
      println(AlphaProba)
      CSB     = CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, BPP.Variables,zeros(BPP.NBvariables),zeros(Int64,0), BPP.LeftMembers_Constraints, zeros(BPP.NBconstraints), zeros(2,BPP.NBvariables), zeros(BPP.NBvariables))
      Stat   = ProbStat(0.0,zeros(Float64,4),typemax(Float64),zeros(Float64,4),zeros(Int64,4),CSB)
      itmax  = 200
      itmax1 = 10
      AlphaValueOBJ = Array{Int64}(4,itmax*itmax1)
      #fill!(AlphaValueOBJ,Vector{Int64})

      for k in 1:1:itmax1
         for j in 1:1:itmax
            CS     = deepcopy(CSB)
            indLa,Alpha    = ReactiveGrasp(AlphaProba,AlphaVal)
            CS             = GraspConstruction(CS,Alpha)
            if CS.CurrentObjectiveValue > Stat.Max
               Stat.Max                   = CS.CurrentObjectiveValue
               Stat.BestSolution          = deepcopy(CS)
            elseif CS.CurrentObjectiveValue < Stat.Min
               Stat.Min                   = CS.CurrentObjectiveValue
            end
            Stat.Sum[indLa]           += CS.CurrentObjectiveValue
            Stat.NBdone[indLa]        += 1
            AlphaValueOBJ[indLa,convert(Int,Stat.NBdone[indLa])]  = CS.CurrentObjectiveValue
         end
         for d in 1:1:4
            Stat.Average[d] = Stat.Sum[d]/Stat.NBdone[d]
         end
         AlphaProba         = UpdateReactiveGrasp(AlphaProba, Stat.Average,Stat.Min,Stat.Max)
      end
      println("After the ",itmax1*itmax," run we got :")
      println(AlphaProba)
      println(AlphaVal)
      println("Maximum found : ",Stat.Max)
      println("Minimum found : ",Stat.Min)
      println("Average : ",Stat.Average)
      println("Number of runs : ",Stat.NBdone)
      println("Grasp construction : ",Stat.BestSolution.CurrentObjectiveValue," with ",CS.CurrentVarUsed)
      HistoryY,SIMUaNNE = SimulatedAnnealing(Stat.BestSolution,500.0,0.95,150,1.0)
      println("Solution with Simulated Annealing : \n",SIMUaNNE.CurrentObjectiveValue)
      HistoryX               = collect(1:1:length(HistoryY))
      #I plot the 911
      title(FileList[i])
      plot(HistoryX,HistoryY, "--")
      xlabel("Iterations")
      ylabel("Objective value")
      grid("on")

      #plot(y,Historyx, "b-", linewidth=2)
      #xlabel("Iterations")
      #ylabel("Objective value")
      #title(FileList[i])
      #Plotting the 911 zith the government
      #=p = bar(AlphaVal,Stat.Max,align="center",alpha=0.4)
      xlabel("Alpha value")
      ylabel("Objective value")
      title("Which Alpha is better ?")
      grid("on")

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
   elseif nbProb >=4
      break;
   end

end

#=
for resu in eachindex(Resume)
   println("For the problem ",Resume[resu].Name," with ",Resume[resu].NBvariables, " variables.")
   println("GLPK Solver Objective value : ",Resume[resu].GLPKObj," Time : ",Resume[resu].GLPKTime ) # getObjectiveValue(model_name) gives the optimum objective value
   println("Heuristics :",Resume[resu].HeurObj," Time : ",Resume[resu].HeurTime,"\n")
end=#
