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
   HeurConsObj::Int
   HeureLSObj::Int
   GLPKTime::Float64
   HeurConsTime::Float64
   HeurLSTime::Float64
   Result() = new()#Allow to create unintialized type
end
type ProbStat
   Max::Float64
   Average::Vector{Float64}
   Min::Float64
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
   if  BPP.NBvariables == 100
      println("Probleme : ",FileList[i])
      ProbTemp = Result();
      CSB    = CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, BPP.Variables,zeros(BPP.NBvariables),zeros(Int64,0), BPP.LeftMembers_Constraints, zeros(BPP.NBconstraints), zeros(2,BPP.NBvariables), zeros(BPP.NBvariables))
      AlphaVal   = [0.5, 0.6, 0.75, 0.9]
      AlphaProba = [0.25,0.25,0.25,0.25]
      println("Before the run we got these Lambda :")
      println(AlphaVal)
      println("With these probabilities :")
      println(AlphaProba)
      Stat   = ProbStat(0.0,zeros(Float64,4),typemax(Float64),zeros(Float64,4),zeros(Float64,4),zeros(Float64,4),zeros(Int64,4),CSB)
      itmax  = 10
      itmax1 = 20
      GraspOBJ = Array{Int64}(itmax*itmax1)
      MaxObj   = Array{Int64}(itmax*itmax1)
      LSOBJ    = Array{Int64}(itmax*itmax1)
      #fill!(AlphaValueOBJ,Vector{Int64})
      @time for k in 1:1:itmax1
         for j in 1:1:itmax
            CS     = deepcopy(CSB)
            indLa,Alpha    = ReactiveGrasp(AlphaProba,AlphaVal)
            Stat.HeurConsTime[indLa] += @elapsed CS = GraspConstruction(CS,Alpha)
            GraspOBJ[j*k]           = CS.CurrentObjectiveValue
            Stat.HeurLSTime[indLa] += @elapsed CS = SimpleGreedyLocalSearch(CS)
            LSOBJ[j*k]              = CS.CurrentObjectiveValue
            if CS.CurrentObjectiveValue > Stat.Max
               Stat.Max                   = CS.CurrentObjectiveValue
               Stat.BestSolution          = deepcopy(CS)
            elseif CS.CurrentObjectiveValue < Stat.Min
               Stat.Min                   = CS.CurrentObjectiveValue
            end
            MaxObj[j*k]                = Stat.Max
            Stat.Sum[indLa]           += CS.CurrentObjectiveValue
            Stat.NBdone[indLa]        += 1
            AlphaValueOBJ[j*k]         = CS.CurrentObjectiveValue
         end
         for d in 1:1:4
            Stat.Average[d] = Stat.Sum[d]/Stat.NBdone[d]
            Stat.HeurConsTime[d] = Stat.HeurConsTime[d]/Stat.NBdone[d]
            Stat.HeurLSTime[d]   = Stat.HeurLSTime[d]/Stat.NBdone[d]
            Stat.HeurConsTime[d] = round(Stat.HeurConsTime[d],5)
            Stat.HeurLSTime[d]   = round(Stat.HeurLSTime[d],5)
         end
         AlphaProba         = UpdateReactiveGrasp(AlphaProba, Stat.Average,Stat.Min,Stat.Max)
      end
      println("After the ",itmax1*itmax," run we got :")
      println(AlphaProba)
      println(AlphaVal)
      println("Maximum found : ",Stat.Max)
      println("Minimum found : ",Stat.Min)
      println("Average : ",Stat.Average)
      println("Average GRASP construction time :",Stat.HeurConsTime)
      println("Average LS time :",Stat.HeurLSTime)
      println("Number of runs : ",Stat.NBdone)
      println("Grasp construction : ",Stat.BestSolution.CurrentObjectiveValue)
      #HistoryX               = collect(1:1:itmax*itmax1)
      #plotRunGrasp(FileList[i],GraspOBJ, LSOBJ, MaxObj)
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
      nbProb+=1
   elseif nbProb >=10
      break;
   end

end
function plotRunGrasp(iname,zinit, zls, zbest)
    figure("Examen d'un run",figsize=(6,6)) # Create a new figure
    title("GRASP-SPP | zConst/zLS/zBest | "iname)
    xlabel("Itérations")
    ylabel("valeurs de z(x)")
    ylim(0, maximum(zbest)+2)

    nPoint = length(zinit)
    x=collect(1:nPoint)
    xticks([1,convert(Int64,ceil(nPoint/4)),convert(Int64,ceil(nPoint/2)), convert(Int64,ceil(nPoint/4*3)),nPoint])
    plot(x,zbest, linewidth=2.0, color="green", label="meilleures solutions")
    plot(x,zls,ls="",marker="^",ms=2,color="green",label="toutes solutions améliorées")
    plot(x,zinit,ls="",marker=".",ms=2,color="red",label="toutes solutions construites")
    vlines(x, zinit, zls, linewidth=0.5)
    legend(loc=4, fontsize ="small")
end
#=
for resu in eachindex(Resume)
   println("For the problem ",Resume[resu].Name," with ",Resume[resu].NBvariables, " variables.")
   println("GLPK Solver Objective value : ",Resume[resu].GLPKObj," Time : ",Resume[resu].GLPKTime ) # getObjectiveValue(model_name) gives the optimum objective value
   println("Heuristics :",Resume[resu].HeurObj," Time : ",Resume[resu].HeurTime,"\n")
end=#
