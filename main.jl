#Ferrari Leon
#M1 ORO
#Version de test
#Julia JuMP
#DM1 - Metaheuristiques
using JuMP, GLPKMathProgInterface,PyPlot

include("myHeuristics.jl")
include("plotting_meta.jl")
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
   CurrentObjectiveValue::Int64
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
   HeurConsObj::Int
   HeureLSObj::Int
   GLPKTime::Float64
   HeurConsTime::Float64
   HeurLSTime::Float64
   Result() = new()#Allow to create unintialized type
end
type ProbStat
   Max::Int64
   Average::Vector{Float64}
   Min::Int64
   NBdone::Vector{Float64}
   HeurConsTime::Vector{Float64}
   HeurLSTime::Vector{Float64}
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
   # FileList[i] == "pb_100rnd0700.dat" || FileList[i] == "pb_100rnd0700.dat"  || FileList[i] == "pb_1000rnd0100.dat" || FileList[i] =="pb_100rnd0100.dat" || FileList[i] == "pb_2000rnd0100.dat"  || FileList[i] == "pb_200rnd0100.dat" || FileList[i] == "pb_500rnd0100.dat"
   # BPP.NBvariables <= 1000 && BPP.NBconstraints <= 500 && nbProb <= 12
#  BPP.NBvariables == 100 && nbProb <10
   if FileList[i] == "pb_100rnd0900.dat"
      println("Probleme : ",FileList[i])
      ProbTemp = Result();
      CSB    = CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, BPP.Variables,zeros(BPP.NBvariables),zeros(Int64,0), BPP.LeftMembers_Constraints, zeros(BPP.NBconstraints), zeros(2,BPP.NBvariables), zeros(BPP.NBvariables))
      AlphaVal   = [0.5, 0.6, 0.75, 0.9]
      AlphaProba = [0.25,0.25,0.25,0.25]
      println("Before the run we got these Lambda :")
      println(AlphaVal)
      println("With these probabilities :")
      println(AlphaProba)
      Stat   = ProbStat(0,zeros(Float64,4),typemax(Int64),zeros(Float64,4),zeros(Float64,4),zeros(Float64,4),zeros(Int64,4),CSB)
      itmax  = 20
      itmax1 = 10
      GraspOBJ = Vector{Int64}(itmax*itmax1)
      MaxObj   = Vector{Int64}(itmax*itmax1)
      LSOBJ    = Vector{Int64}(itmax*itmax1)
      zmaxit   = Vector{Int64}(itmax1)
      zminit   = Vector{Int64}(itmax1)
      zmoyit   = Vector{Int64}(itmax1)
      avgttot = 0
      #fill!(AlphaValueOBJ,Vector{Int64})
      @time for k in 1:1:itmax1
         for j in 1:1:itmax
            currIndex = ((k-1)*20)+j
            CS     = deepcopy(CSB)

            indLa,Alpha    = ReactiveGrasp(AlphaProba,AlphaVal)
            Stat.HeurConsTime[indLa] += @elapsed CS = GraspConstruction(CS,Alpha)
            GraspOBJ[currIndex]           = CS.CurrentObjectiveValue

            Stat.HeurLSTime[indLa] += @elapsed CS = SimpleGreedyLocalSearch(CS)
            #Stat.HeurLSTime[indLa] += @elapsed CS = SimulatedAnnealing(CS,500.0,0.95,100,1.0)
            LSOBJ[currIndex]              = CS.CurrentObjectiveValue

            if CS.CurrentObjectiveValue > Stat.Max
               Stat.Max                   = CS.CurrentObjectiveValue
               Stat.BestSolution          = deepcopy(CS)
            elseif CS.CurrentObjectiveValue < Stat.Min
               Stat.Min                   = CS.CurrentObjectiveValue
            end

            MaxObj[currIndex]                = Stat.Max
            Stat.Sum[indLa]           += CS.CurrentObjectiveValue
            Stat.NBdone[indLa]        += 1
            avgttot += CS.CurrentObjectiveValue
         end
         for d in 1:1:4
            Stat.Average[d] = Stat.Sum[d]/Stat.NBdone[d]
            Stat.HeurConsTime[d] = Stat.HeurConsTime[d]/Stat.NBdone[d]
            Stat.HeurLSTime[d]   = Stat.HeurLSTime[d]/Stat.NBdone[d]
            Stat.HeurConsTime[d] = round(Stat.HeurConsTime[d],5)
            Stat.HeurLSTime[d]   = round(Stat.HeurLSTime[d],5)

         end
         zmaxit[k] = Stat.Max
         zminit[k] = Stat.Min
         zmoyit[k] = convert(Int64, floor(avgttot /(k*20)))
         println("After the ",k*20," run we got :")
         println("Maximum found : ",Stat.Max)
         println("Minimum found : ",Stat.Min)
         println("Average : ",Stat.Average)
         AlphaProba         = UpdateReactiveGrasp(AlphaProba, Stat.Average,Stat.Min,Stat.Max)
      end
      #=println("After the ",itmax1*itmax," run we got :")
      println(AlphaProba)
      println(AlphaVal)
      println("Maximum found : ",Stat.Max)
      println("Minimum found : ",Stat.Min)
      println("Average : ",Stat.Average)
      println("Average GRASP construction time :",Stat.HeurConsTime)
      println("Average LS time :",Stat.HeurLSTime)
      println("Number of runs : ",Stat.NBdone)
      println("Grasp construction : ",Stat.BestSolution.CurrentObjectiveValue)=#
      HistoryX  = Vector{Int64}(itmax1)
      for div = 1:itmax1
         HistoryX[div]  =  itmax * div
      end
      #plotRunGrasp(FileList[i],GraspOBJ, LSOBJ, MaxObj)
      plotAnalyseGrasp(FileList[i],HistoryX,zmoyit,zminit,zmaxit)
#=
      @variable(   m , x[1:BPP.NBvariables], Bin)
      @objective(  m , Max, sum( BPP.Variables[j] * x[j] for j=1:BPP.NBvariables ) )
      @constraint( m , cte[i=1:BPP.NBconstraints], sum(BPP.LeftMembers_Constraints[i,j] * x[j] for j=1:BPP.NBvariables) <= BPP.RightMembers_Constraints[i] )
      #SOLVE IT AND DISPLAY THE RESULTS
      #--------------------------------
      ProbTemp.GLPKTime       = @elapsed solve(m) # solves the model
      ProbTemp.NBvariables    = BPP.NBvariables
      ProbTemp.Name           = string("./Data/",FileList[i])
      ProbTemp.GLPKObj        = getobjectivevalue(m)
      ProbTemp.HeurConsObj    = CS.CurrentObjectiveValue
      Resume[nbProb]          = deepcopy(ProbTemp)=#
      nbProb=10
   elseif nbProb >=10
      break;
   end

end

#=
for resu in eachindex(Resume)
   println("For the problem ",Resume[resu].Name," with ",Resume[resu].NBvariables, " variables.")
   println("GLPK Solver Objective value : ",Resume[resu].GLPKObj," Time : ",Resume[resu].GLPKTime ) # getObjectiveValue(model_name) gives the optimum objective value
   println("Heuristics :",Resume[resu].HeurObj," Time : ",Resume[resu].HeurTime,"\n")
end=#
