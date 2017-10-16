#Algorithme de descente glouton
#Principe de cet algorithme :
#Creation d'un heuristique de construction permettant de trouver solution admissible x0
#cette heuristique de depart sera develloper ici
#Mise en place d'une heuristique de recherche locale de type plus profonde descente
#Celle ci sera fondee sur deux voisins
#Order variable by decreasing cost
#Align the constraints with the variable order
#Compute the ratio between the cost and the total number of occurence of the variable in all the constraints
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
#LastRightMemberValue sera un vecteur contenant la derniere valeur calculee pour la somme des membres de gauche des contraintes
#LastModifiedIndex sera un entier ayant pour valeur l'indice de la derniere variable modifie de 0 a 1
#LastRightMemberValue est la matrice des contraintes "Actualisee" ou si la variable Xj a pour valeur 1, les lignes de la matrice ou Xj est present seront passee a 0 et inversement
function GraspConstruction(CS::CurrentSolution, Alpha::Float64)
   CS                   = ComputeUtility(CS)
   cs1                  = CS
   Available            = CS.NBvariables
   Utilities            = deepcopy(CS.Utility)
   RandomCandidateList  = Int[]
	#println(CS.Utility)
   while Available > 1
      AboveTheLimit        = 0
      #CS.Utility save the real utility array at t = 1
      #Utilities is the current utilies, so without the impossible variables
      RandomCandidateList  = empty!(RandomCandidateList)
      LimitSelect = (minimum(Utilities[2,:]) + (Alpha * (maximum(Utilities[2,:])-minimum(Utilities[2,:]))))
		#println(minimum(Utilities[2,:]), " + (" ,Alpha, " * " ,"(",maximum(Utilities[2,:]),- ,minimum(Utilities[2,:]),"))")
      #println("Limit for utility is :",LimitSelect)
      for i = 1:1:Available
         if Utilities[2,i] >= LimitSelect
            RandomCandidateList = push!(RandomCandidateList,Utilities[1,i])
            AboveTheLimit       += 1
         end
      end
		#println("Available : ",Available, " AboveTheLimit : ",AboveTheLimit)
      if AboveTheLimit == 0
         break
      else
         #println("Candidate list is :",RandomCandidateList)
         RandomPickedCandidate   = rand(RandomCandidateList)
         #println("We picked ",RandomPickedCandidate)
         answer,cs1              = SetToOne(cs1,RandomPickedCandidate)
			if answer
				#println("Current Solution : ",cs1.CurrentVariables,"\n For an OBJ worth :",cs1.CurrentObjectiveValue)
			else
				println("Failed")
			end

         Available,Utilities     = UpdateUtility(cs1)
      end
   end
   #println("GRASP construction with ",Alpha," for lambda  : ",cs1.CurrentObjectiveValue)
   return cs1
end

function ComputeUtility(CS::CurrentSolution)
	max = 0
   for i = 1:1:CS.NBvariables
         CS.Utility[1,i]   = i
         nb                = sum(CS.LeftMembers_Constraints[:,i])
			if nb == 0
				CS.Utility[2,i]   = 0
			else
				CS.Utility[2,i]   = CS.Variables[i]/nb
				if CS.Utility[2,i]> max
					max = CS.Utility[2,i]
				end
			end
   end
	for i in eachindex(CS.Variables)
		if CS.Utility[2,i] == 0
			CS.Utility[2,i]   = max
		end
	end

   CS.Utility    = sortcols(CS.Utility, rev=true, by = x -> (x[2]))
   return CS
end

function UpdateUtility(CS::CurrentSolution)
   UtilitiesIndex    = Float64[]
   UtilitiesValues   = Float64[]
   Inc               = 1
   for i = 1:1:CS.NBvariables
		index = convert(Int,CS.Utility[1,i])
      if CS.Freedom[index] == 0 && CS.CurrentVariables[index] == 0
         UtilitiesIndex    = push!(UtilitiesIndex,CS.Utility[1,i])
         UtilitiesValues   = push!(UtilitiesValues,CS.Utility[2,i])
         Inc += 1
      end
   end
   Utilities         = Matrix(0,Inc-1)
   Utilities         = vcat(Utilities,UtilitiesIndex')
   Utilities         = vcat(Utilities,UtilitiesValues')
   return Inc-1,Utilities
end

#Fonction recursive permettant l'eploration des solutions voisines admissibles
function LocalSearch(CS::CurrentSolution, Randomness::Int)
   OldObjectiveValue = CS.CurrentObjectiveValue
   FailedToImprove   = 0
   CurrentVarUsed    = Int64[]
   CurrentBestSol    = 0
   CS_arr= Vector{CurrentSolution}(Randomness)
   while FailedToImprove < (CS.NBvariables)
      for cv in eachindex(CS.CurrentVariables)
         if CS.CurrentVariables[cv] == 1
            push!(CurrentVarUsed,cv)
         end
      end
      CurrentBestSol = 0
      RandomlyPickedVar=union(rand(CurrentVarUsed,Randomness-1))
      fill!(CS_arr,CS)
      for i = 1:(length(RandomlyPickedVar)-1)#retirer 1 a la taille
         CS_arr[i]            = deepcopy(CS)
         answerz,CS_arr[i]    = SetToZero(CS_arr[i],RandomlyPickedVar[i])
         if answerz
            CS_arr[Randomness]   = CS_arr[i]

            for j = 1:1:CS.NBvariables
               if CS_arr[i].Utility[1,j] != RandomlyPickedVar[i]
                  # previous condition : && CS_arr[i].Freedom[convert(Int,CS_arr[i].Utility[1,j])] == 0
                  answer,CS_arr[i]   =  SetToOne(CS_arr[i],convert(Int64,CS_arr[i].Utility[1,j]))
                  if answer
                     CS_arr[Randomness]   = CS_arr[i]
                  else
                     CS_arr[i]   = CS_arr[Randomness]
                  end
               end
            end
         end

      end
      #Garde la meilleur valeur objective et l'index de la meilleure solution
      for sol  = 1:1:length(CS_arr)
         if CS_arr[sol].CurrentObjectiveValue > OldObjectiveValue
               CS = deepcopy(CS_arr[sol])
               #println("We improved it !",OldObjectiveValue,"-->",CS.CurrentObjectiveValue)
               OldObjectiveValue = CS_arr[sol].CurrentObjectiveValue
               CurrentBestSol=1
         end
      end

      if CurrentBestSol == 0
         FailedToImprove += 1
      else
         FailedToImprove = 0
      end
   end
   return CS
end

function SetToZero(CS::CurrentSolution, x::Int)
   if CS.CurrentVariables[x] == 1
      for j in 1:1:CS.NBconstraints
            if CS.LeftMembers_Constraints[j,x] == 1
               CS.LastRightMemberValue_Constraint[j] = 0
               for i in 1:1:CS.NBvariables
                  if CS.LeftMembers_Constraints[j,i] == 1
                     CS.Freedom[i]+=1
                  end
               end
            end
      end
   else
      return false,CS
   end
   CS.CurrentVariables[x] = 0
   CS.CurrentObjectiveValue-=CS.Variables[x]
   return true,CS
end

function SetToOne(CS::CurrentSolution, x::Int)
   if CS.Freedom[x] == 0 && CS.CurrentVariables[x] == 0
      for j in 1:1:CS.NBconstraints
            if CS.LeftMembers_Constraints[j,x] == 1
               if CS.LastRightMemberValue_Constraint[j] ==  0
                  CS.LastRightMemberValue_Constraint[j] = 1
                  for i in 1:1:CS.NBvariables
                     if CS.LeftMembers_Constraints[j,i] == 1
                        CS.Freedom[i]-=1
                     end
                  end

                  # X = la matrice des contraites
                  # V = LastLeftMemberValue_Constraint
                  # Fixe V(i,j) = 0 si X(i,j) ==1 dans la matrice des contraines et si la variable Xj a pour valeur 1
                  # sinon V(i,j)=1 si X(i,j) ==1 dans la matrice des contraitnes et si la Variable Xj a pour valeur 0
               else
                  #println("Setting one to X",x," violate the constraint number ",j,".")
                  return false,CS
               end
            end
      end
   else
      return false,CS
   end
   CS.CurrentVariables[x] = 1
   CS.CurrentObjectiveValue+=CS.Variables[x]
   #println("Variable X",x," have been set to one.")
   return true,CS
end


function UpdateReactiveGrasp(AlphaProba::Vector{Float64},Average::Vector{Float64},Worst::Vector{Float64},Max::Vector{Float64})
   NewValue = Vector{Float64}(length(AlphaProba))
   for i in eachindex(AlphaProba)
      #println("( ",Average[i]," - ",Worst[i]," ) /( ",Max[i]," - ",Worst[i], " )")
      #Wrong ig Average or worst are the same value or if max and worst are the same value
      NewValue[i] = ( Average[i] - Worst[i] ) / ( Max[i] - Worst[i] )
   end
   SumOfNew = sum(NewValue)
   for i in eachindex(AlphaProba)
      AlphaProba[i] = NewValue[i] / SumOfNew
   end
   return AlphaProba
end

function ReactiveGrasp(AlphaProba::Vector{Float64},AlphaVal::Vector{Float64})
   Proba = rand()
   Val = 0
   for i in eachindex(AlphaProba)
      Val += AlphaProba[i]
      if Proba <= Val
         return i,AlphaVal[i]
      end
   end
   return length(AlphaVal),AlphaVal[length(AlphaVal)]
end

function SimulatedAnnealing(CS::CurrentSolution,InitTemperature::Int,CoolingCoef::Float64,StepSize::Int,MinTemp::Float64)
   CSTemp      = deepcopy(CS)
   CSBest      = deepcopy(CS)
   Temperature = convert(Float64,InitTemperature)
   LOL         = true
	Historic 	= Int[]
   while LOL
      for i in 1:1:StepSize
         LocalCS 			= GetRandomNeighbour(CSTemp)
         #LocalCS        = deepcopy(CSTemp) #Work on CSTemp
         DeltaObj       = LocalCS.CurrentObjectiveValue - CSTemp.CurrentObjectiveValue
			ValueOf			= exp(DeltaObj/Temperature)
			RandValue 		= rand()
         if DeltaObj > 0 || ValueOf > RandValue
				#println("Solution accepted : f(x) ",CSTemp.CurrentObjectiveValue, " --> "," f'(x) : ",CSTemp.CurrentObjectiveValue)
            CSTemp      = deepcopy(LocalCS)
				Historic		= push!(Historic,CSTemp.CurrentObjectiveValue)
            if CSTemp.CurrentObjectiveValue > CSBest.CurrentObjectiveValue
               CSBest   =  deepcopy(CSTemp)
					println("Improved ! We got : ",CSBest.CurrentObjectiveValue)
            end
         end
      end
      Temperature *= CoolingCoef
      if Temperature < MinTemp
         LOL = false
      end
   end
   return Historic,CSBest
end


#THis function just do a 1/1 swap
#Must be changed to AddMultipleOrElseDrop1
function GetRandomNeighbour(CS::CurrentSolution)
   CurrentVarUsed    = Int64[]

   for VarOn in eachindex(CS.CurrentVariables)
      if CS.CurrentVariables[VarOn] == 1
         push!(CurrentVarUsed,VarOn)
      end
   end

   while true
      CurrentVarFree    = Int64[]
      RandomlyPickedUsedVar=rand(CurrentVarUsed)
      CSTemp            = deepcopy(CS)
      answerz,CSRand    = SetToZero(CSTemp,RandomlyPickedUsedVar)

      if answerz
         CSTemp            = deepcopy(CSRand)
         for j = 1:1:CS.NBvariables
            index = convert(Int64,CSRand.Utility[1,j])
            if index != RandomlyPickedUsedVar && CSRand.Freedom[index] == 0
               answer,CSTemp   =  SetToOne(CSTemp,index)
               if answer
                  #println("Old solution value : ",CS.CurrentObjectiveValue,"\n After setting x",RandomlyPickedUsedVar," to 0 we have  ",CSRand.CurrentObjectiveValue)
                  #println("After setting x",RandomlyPickedFreeVar," to 1 we got :",CSTemp.CurrentObjectiveValue)
                  CSRand   = deepcopy(CSTemp)
               else
                  CSTemp   = deepcopy(CSRand)
               end
               #push!(CurrentVarFree,convert(Int64,CSRand.Utility[1,j]))
            end
         end
         return CSRand
         #=if length(CurrentVarFree)>0
            RandomlyPickedFreeVar = rand(CurrentVarFree)
            answer,CSTemp   =  SetToOne(CSTemp,RandomlyPickedFreeVar)
            if answer
               #println("Old solution value : ",CS.CurrentObjectiveValue,"\n After setting x",RandomlyPickedUsedVar," to 0 we have  ",CSRand.CurrentObjectiveValue)
               #println("After setting x",RandomlyPickedFreeVar," to 1 we got :",CSTemp.CurrentObjectiveValue)
               return CSTemp
            else
               CSTemp   = CSRand
            end
         end=#
      end
   end
   return CS
end
