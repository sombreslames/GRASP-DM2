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
   Sum::Vector{Int64}
   BestSolution::Vector{CurrentSolution}
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
   if nbProb <= 4 && BPP.NBvariables <= 100 && BPP.NBconstraints <= 400
      println("Probleme : ",FileList[i])
      #ProbTemp = Result();

      AlphaVal   = [0.5, 0.6, 0.75, 0.9]
      AlphaProba = [0.25,0.25,0.25,0.25]

      #println("Before the run we got these Lambda :")
      #println(AlphaVal)
      #println("With these probabilities :")
      #println(AlphaProba)
      #Stat        = ProbStat(zeros(Float64,4),zeros(Float64,4),zeros(Float64,4),zeros(Float64,4),zeros(Int64,4),Vector{CurrentSolution}(4))
      #fill!(Stat.Min, typemax(Float64))
      itmax  = 1000
      itmax1 = 30
      AlphaValueOBJ = Array{Int64}(4,itmax*itmax1)
      #fill!(AlphaValueOBJ,Vector{Int64})

      #=for k in 1:1:itmax1
         for j in 1:1:itmax
            indLa,Alpha    = ReactiveGrasp(AlphaProba,AlphaVal)
            CS             = GraspConstruction(CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, BPP.Variables,zeros(BPP.NBvariables), BPP.LeftMembers_Constraints, zeros(BPP.NBconstraints), zeros(2,BPP.NBvariables), zeros(BPP.NBvariables)),Alpha)
            #println(j);
            #CS             = SimulatedAnnealing(CS,500,0.75,5,10)
            #test           = GetRandomNeighbour(CS)
            #println(test.CurrentObjectiveValue)
            #Effectuer la recherche locale ici
            #CS             = LocalSearch(CS,convert(Int,Alpha*CS.NBvariables))
            if CS.CurrentObjectiveValue > Stat.Max[indLa]
               Stat.Max[indLa]            = CS.CurrentObjectiveValue
               Stat.BestSolution[indLa]   = deepcopy(CS)
            elseif CS.CurrentObjectiveValue < Stat.Min[indLa]
               Stat.Min[indLa]            = CS.CurrentObjectiveValue
            end
            Stat.Sum[indLa]           += CS.CurrentObjectiveValue
            Stat.NBdone[indLa]        += 1
            AlphaValueOBJ[indLa,convert(Int,Stat.NBdone[indLa])]  = CS.CurrentObjectiveValue
         end
         for d in 1:1:4
            Stat.Average[d] = Stat.Sum[d]/Stat.NBdone[d]
         end
         #GetRandomNeighbour(Stat.BestSolution[1])

         AlphaProba         = UpdateReactiveGrasp(AlphaProba, Stat.Average,Stat.Min,Stat.Max)

         #SIMUaNNE = SimulatedAnnealing(Stat.BestSolution[1],500,0.75,5,10)
         #println("Solution with Simulated Annealing : \n",SIMUaNNE.CurrentObjectiveValue)
      end
      println("After the ",itmax1*itmax," run we got :")
      println(AlphaProba)
      println(AlphaVal)
      println("Maximum found : ",Stat.Max)
      println("Minimum found : ",Stat.Min)
      println("Average : ",Stat.Average)
      println("Number of runs : ",Stat.NBdone)=#
      indLa,Alpha    = ReactiveGrasp(AlphaProba,AlphaVal)
      CS             = GraspConstruction(CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, BPP.Variables,zeros(BPP.NBvariables), BPP.LeftMembers_Constraints, zeros(BPP.NBconstraints), zeros(2,BPP.NBvariables), zeros(BPP.NBvariables)),Alpha)
      println("Got :",CS.CurrentObjectiveValue, " With GRASP construction")
      HistoryY,SIMUaNNE = SimulatedAnnealing(CS,500,0.95,convert(Int,1.5*CS.NBvariables),1)
      println("Solution with Simulated Annealing : \n",SIMUaNNE.CurrentObjectiveValue)
      HistoryX               = collect(1:1:length(HistoryY))
      println("y :",length(HistoryY),"x :",length(HistoryX))

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
   elseif nbProb >4
      break;
   end

end

#=
for resu in eachindex(Resume)
   println("For the problem ",Resume[resu].Name," with ",Resume[resu].NBvariables, " variables.")
   println("GLPK Solver Objective value : ",Resume[resu].GLPKObj," Time : ",Resume[resu].GLPKTime ) # getObjectiveValue(model_name) gives the optimum objective value
   println("Heuristics :",Resume[resu].HeurObj," Time : ",Resume[resu].HeurTime,"\n")
end=#
