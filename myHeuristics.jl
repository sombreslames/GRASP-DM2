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
#Ifg we order the utility we can stop directly after one var is under the limit
function GraspConstruction(CS::CurrentSolution, Alpha::Float64)
   Available,CS         = ComputeUtilityPlus(CS)
   cs1                  = deepcopy(CS)
   RandomCandidateList  = Int[]
   while Available >= 1
      AboveTheLimit        = 0
      RandomCandidateList  = empty!(RandomCandidateList)
      LimitSelect = (minimum(cs1.Utility[2,:]) + (Alpha * (maximum(cs1.Utility[2,:])-minimum(cs1.Utility[2,:]))))
      for i = 1:1:Available
         if cs1.Utility[2,i] >= LimitSelect
            RandomCandidateList = push!(RandomCandidateList,cs1.Utility[1,i])
            AboveTheLimit       += 1
         else
            break
         end
      end

      if AboveTheLimit == 0
         break
      else
         RandomPickedCandidate   = rand(RandomCandidateList)
         answer,cs1              = SetToOne(cs1,RandomPickedCandidate)
			if answer == false
				println("Failed")
			end
      end
      Available,cs1.Utility     = UpdateUtility(cs1)
      #println("There are still ",Available," var available")
   end
   cs1.Utility = deepcopy(CS.Utility)
   return cs1
end
function ComputeUtilityPlus(CS::CurrentSolution)
   UtilitiesIndex    = Float64[]
   UtilitiesValues   = Float64[]
   Inc               = 1
   for i = 1:1:CS.NBvariables
      nb                = sum(CS.LeftMembers_Constraints[:,i])
      if nb != 0
         UtilitiesIndex    = push!(UtilitiesIndex,i)
         UtilitiesValues   = push!(UtilitiesValues,(CS.Variables[i]/nb))
         Inc += 1
      else
         CS.CurrentVariables[i]   = 2
         CS.Freedom[i]            = 2
         CS.CurrentObjectiveValue += CS.Variables[i]
         #println("x",i," is not in any constraint we set it to 1. And Kick it from the problem")
      end
   end
   Utilities         = Matrix(0,Inc-1)
   Utilities         = vcat(Utilities,UtilitiesIndex')
   Utilities         = vcat(Utilities,UtilitiesValues')
   CS.Utility        = deepcopy(Utilities)
   CS.Utility        = sortcols(CS.Utility, rev=true, by = x -> x[2])
   return Inc-1,CS
end

function UpdateUtility(CS::CurrentSolution)
   UtilitiesIndex    = Float64[]
   UtilitiesValues   = Float64[]
   Inc               = 1
   size              = length(CS.Utility[1,:])
   for i = 1:1:size
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
   CS.CurrentVarUsed          = deleteat!(CS.CurrentVarUsed,findin(CS.CurrentVarUsed,x))
   CS.CurrentVariables[x]     = 0
   CS.CurrentObjectiveValue   -=CS.Variables[x]
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
               else
                  return false,CS
               end
            end
      end
   else
      return false,CS
   end
   CS.CurrentVarUsed       = push!(CS.CurrentVarUsed,x)
   CS.CurrentVariables[x]  = 1
   CS.CurrentObjectiveValue+=CS.Variables[x]
   return true,CS
end


function UpdateReactiveGrasp(AlphaProba::Vector{Float64},Average::Vector{Float64},Worst::Int64,Max::Int64)
   NewValue = Vector{Float64}(length(AlphaProba))
   for i in eachindex(AlphaProba)
      NewValue[i] = ( Average[i] - Worst ) / ( Max - Worst )
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
#Rules 1 : You will now call GRASP the cycle of foundation
#Rules 2 : No computer sciences student can control others people feelings
#Rules 3 : Waiting for climate change, after all URSS invented it.
function SimulatedAnnealing(CS::CurrentSolution,InitTemperature::Float64,CoolingCoef::Float64,StepSize::Int,MinTemp::Float64)
   CSTemp      = deepcopy(CS)
   CSBest      = deepcopy(CS)
   Temperature = InitTemperature
   ClimateChange         = true
	#Historic 	= Int[]
   nbRun       = 0
   while ClimateChange
      for i in 1:1:StepSize
         LocalCS        = AddOrElseDrop(CSTemp)
         #LocalCS 			= GetRandomNeighbour(CSTemp)
         DeltaObj       = LocalCS.CurrentObjectiveValue - CSTemp.CurrentObjectiveValue
			ValueOf			= exp(DeltaObj/Temperature)
			RandValue 		= rand()
         if DeltaObj > 0 || ValueOf > RandValue
				#println("Solution accepted : f(x) ",CSTemp.CurrentObjectiveValue, " --> "," f'(x) : ",CSTemp.CurrentObjectiveValue)
            CSTemp      = deepcopy(LocalCS)
				#Historic		= push!(Historic,CSTemp.CurrentObjectiveValue)
            if CSTemp.CurrentObjectiveValue > CSBest.CurrentObjectiveValue
               CSBest   =  deepcopy(CSTemp)
					#println("Improved ! We got : ",CSBest.CurrentObjectiveValue)
            end
         end
      end
      nbRun += 1
      #println("Nb of run : ",StepSize * nbRun)
      Temperature *= CoolingCoef
      if Temperature < MinTemp
         ClimateChange = false
      end
   end
   return CSBest
end


function GetRandomNeighbour(CS::CurrentSolution)
   while true
      CurrentVarFree        = Int64[]
      RandomlyPickedUsedVar = rand(CS.CurrentVarUsed)
      #println("We picked ",RandomlyPickedUsedVar," from the used var ",CS.CurrentVarUsed)
      CSTemp                = deepcopy(CS)
      answerz,CSRand        = SetToZero(CSTemp,RandomlyPickedUsedVar)
      if answerz
         CSTemp            = deepcopy(CSRand)
         for j = 1:1:length(CS.Utility[1,:])
            index = convert(Int64,CSRand.Utility[1,j])
            if index != RandomlyPickedUsedVar && CSRand.Freedom[index] == 0
               answer,CSTemp   =  SetToOne(CSTemp,index)
               if answer
                  if CSTemp.CurrentObjectiveValue > CSRand.CurrentObjectiveValue
                     CSRand   = deepcopy(CSTemp)
                  end
               else
                  CSTemp   = deepcopy(CSRand)
               end
            end
         end
         return CSRand
      end
   end
   return CS
end

function AddOrElseDrop(CS::CurrentSolution)
   nb,FreeVar     = UpdateUtility(CS)
   CSTemp         = deepcopy(CS)
   if nb > 0
      RandomVar       = convert(Int,FreeVar[1,rand(1:end)])
      answer,CSTemp   =  SetToOne(CSTemp,convert(Int,RandomVar))
      if answer
         return CSTemp
      else
         println("Duck Duck Duck")
      end
   else
      RandomlyPickedUsedVar = rand(CS.CurrentVarUsed)
      answerz,CSTemp        = SetToZero(CSTemp,RandomlyPickedUsedVar)
      if answerz
         return CSTemp
      else
         println("Damn damn damn")
      end
   end
   return nothing
end
#Un petit N puissance 4 au calme coder en 5 min because no need to opti bro
#meme plus cest du caca
function SimpleGreedyLocalSearch(CS::CurrentSolution)
   TempSolBest = deepcopy(CS)
   for ik = 1:1:length(CS.CurrentVarUsed)
      TempSol1 = deepcopy(CS)
      indexk1 = convert(Int,CS.CurrentVarUsed[ik])
      nothing,TempSol1           = SetToZero(TempSol1,indexk1)
      nothing,TempSol1.Utility   = UpdateUtility(TempSol1)
      for jk = 1:1:length(TempSol1.CurrentVarUsed)
         indexk2 = convert(Int,TempSol1.CurrentVarUsed[jk])
         TempSol2                   = deepcopy(TempSol1)
         nothing,TempSol2           = SetToZero(TempSol2,indexk2)
         nothing,TempSol2.Utility   = UpdateUtility(TempSol2)
         TotDiff = CS.CurrentObjectiveValue - TempSol2.CurrentObjectiveValue
         for ip = 1:1:length(TempSol2.Utility[1,:])
            indexP1 = convert(Int,TempSol2.Utility[1,ip])
            if indexP1 != indexk1 && indexP1 != indexk2
               if TempSol2.Variables[indexP1] > (TotDiff/2)
                  #println(TempSol2.Variables[indexP1]," --> ", TotDiff/2)
                  TempSol3       = deepcopy(TempSol2)
                  HalfDiff       = TotDiff-TempSol2.Utility[2,ip]
                  nothing,TempSol3 = SetToOne(TempSol3,indexP1)
                  nothing,TempSol3.Utility = UpdateUtility(TempSol3)
                  for jp = ip:1:length(TempSol3.Utility[1,:])
                     indexP2 = convert(Int,TempSol3.Utility[1,jp])
                     if  indexP2 != indexk1 && indexP2 != indexk2
                        if TempSol3.Variables[indexP2] > HalfDiff
                           TempSol4          = deepcopy(TempSol3)
                           nothing ,TempSol4 = SetToOne(TempSol4,indexP2)
                           if TempSol4.CurrentObjectiveValue > TempSolBest.CurrentObjectiveValue
                              println(TempSol4.CurrentObjectiveValue, " --> ", TempSolBest.CurrentObjectiveValue)
                              TempSolBest = deepcopy(TempSol4)

                           end
                        end
                     end
                  end
               end
            end
         end
      end
   end
   return TempSolBest
end
