#Ferrari Leon
#M1 ORO
#Version de test
#Julia JuMP
#DM1 - Metaheuristiques
using JuMP, GLPKMathProgInterface, PyPlot
include("myHeuristics.jl")
type Problem
   NBvariables::Int
   NBconstraints::Int
   Variables
   LeftMembers_Constraints
   RightMembers_Constraints
end
type CurrentSolution
   NBconstraints::Int
   NBvariables::Int
   CurrentVarIndex::Int
   CurrentObjectiveValue::Int
   Variables
   CurrentVariables
   LeftMembers_Constraints
   LastRightMemberValue_Constraint
   Utility
   LastLeftMemberValue_Constraint
end

function ReadFile(FileName)
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
            #parcours chaque valeurs (chaque valeur est separer par un espace)
            #ecrire dfct get value qui renvoie un tableau
            #lire dabord la pemiere ligne
            #puis lire le reste du fichier grace au infos obtenu dans la premiere ligne
         end
   end
   close(workingfile)
   return Problem(NBvar, NBcons, Coef, LeftMembers_Constraints, RightMembers_Constraints)
end

#GETTING DATA FRON FILE
FileList = readdir("./Data")
dir = pwd()
x_cpuTime = Vector(length(FileList))
y_problemSize = Vector(length(FileList))
z_problemConstraints = Vector(length(FileList))
for i in eachindex(FileList)
   #MODEL CONSTRUCTION
   #--------------------
   m           = Model(solver=GLPKSolverMIP())
   #READING DATA FROM FILE
   #     GETTING NB OF VARIABLES AND CONSTRAINTS
   #     AND VALUES ASSOCIATED TO THEM
   BPP = ReadFile(string("./Data/",FileList[i]))
   if BPP.NBvariables < 1000 && BPP.NBconstraints < 1000
      tic()
      cs = CurrentSolution(BPP.NBconstraints, BPP.NBvariables, 0, 0, BPP.Variables,Vector(BPP.NBvariables), BPP.LeftMembers_Constraints, Vector(BPP.NBconstraints), zeros(2,BPP.NBvariables), BPP.LeftMembers_Constraints)
      FindingAdmissingBaseSolution1(cs)
      #@variable(  m,  0 <= x[1:BPP.NBvariables] <= 1,Int)
      #=@variable(  m,  x[1:BPP.NBvariables], Bin)
      @objective( m , Max, sum( BPP.Variables[j] * x[j] for j=1:BPP.NBvariables ) )
      @constraint( m , cte[i=1:BPP.NBconstraints], sum(BPP.LeftMembers_Constraints[i,j] * x[j] for j=1:BPP.NBvariables) <= BPP.RightMembers_Constraints[i] )
      #@constraint(m, dot(LeftMembers_Constraints, x) <= RightMembers_Constraints)
      #println("The optimization problem to be solved is:")
      #print(m) # Shows the model constructed in a human-readable form

      #SOLVE IT AND DISPLAY THE RESULTS
      #--------------------------------
      status = solve(m) # solves the model
      x_cpuTime[i]            =  toc()
      y_problemSize[i]        =  BPP.NBvariables
      z_problemConstraints[i] =  BPP.NBconstraints
      println("Objective value: ", getobjectivevalue(m)) # getObjectiveValue(model_name) gives the optimum objective value
      =#
      quit()
   end
end
