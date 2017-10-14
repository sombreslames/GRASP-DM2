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
   CurrentVarIndex::Int
   CurrentObjectiveValue::Int
   Variables
   CurrentVariables
   LeftMembers_Constraints
   LastRightMemberValue_Constraint
   Utility
   LastLeftMemberValue_Constraint
end
#LastRightMemberValue sera un vecteur contenant la derniere valeur calculee pour la somme des membres de gauche des contraintes
#LastModifiedIndex sera un entier ayant pour valeur l'indice de la derniere variable modifie de 0 a 1
#LastRightMemberValue est la matrice des contraintes "Actualisee" ou si la variable Xj a pour valeur 1, les lignes de la matrice ou Xj est present seront passee a 0 et inversement
function FindingAdmissingBaseSolution1(CS::CurrentSolution)
   println("Il y a ",CS.NBvariables," variables")
   #=for i = 1:CS.NBvariables
      CS.Utility[1,i]   = i
      nb                = sum(CS.LeftMembers_Constraints[:,i])
      CS.Utility[2,i]   = CS.Variables[i]/nb
   end=#
   CS    =     UpdateUtility(CS)
   #sortrows(CS.Utility,by=y->y[2])
   #sortcols(Ratio)
   println(CS.Utility)
   return CS
end
#Fonction recursive permettant l'eploration des solutions voisines admissibles
function LocalSearch(CS::CurrentSolution)

end
function UpdateUtility(CS::CurrentSolution)
   for i = 1:CS.NBvariables
      CS.Utility[1,i]   = i
      nb                = sum(CS.LastLeftMemberValue_Constraint[:,i])
      if nb == 0
         CS.Utility[2,i]=0
      else
         CS.Utility[2,i]   = CS.Variables[i]/nb
      end
   end
   sortrows(CS.Utility,by=y->y[2])
   return CS
end

# swap value of x and y
#x=1 --> x=0 and y=0 -- y=1
function Swap2Var(CS::CurrentSolution, x::Int ,y::Int)
   cs1         = CS
   cs1         = SetToZero(cs1,x)
   answer,cs1  = SetToOne(cs1,y)
   if answer && cs1.CurrentObjectiveValue >= CS.CurrentObjectiveValue
      cs1 = UpdateUtility(cs1)
      return true,cs1
   end
   return false,CS
end

function SetToZero(CS::CurrentSolution, x::Int)
   for j in 1:1:CS.NBconstraints
         if CS.LeftMembers_Constraints[j][x] == 1
            CS.LastRightMemberValue_Constraint[j] = 0
            CS.LastLeftMemberValue_Constraint[j,x] = 1
         end
   end
   return CurrentSolution
end
function SetToOne(CS::CurrentSolution, x::Int)
   for j in 1:1:CS.NBconstraints
         if CS.LeftMembers_Constraints[j][x] == 1
            if CS.LastRightMemberValue_Constraint[j] ==  0
               CS.LastRightMemberValue_Constraint[j] = 1
               CS.LastLeftMemberValue_Constraint[j,x] = 0
               # X = la matrice des contraites
               # V = LastLeftMemberValue_Constraint
               # Fixe V(i,j) = 0 si X(i,j) ==1 dans la matrice des contraines et si la variable Xj a pour valeur 1
               # sinon V(i,j)=1 si X(i,j) ==1 dans la matrice des contraitnes et si la Variable Xj a pour valeur 0
            else
               return false,nothing
            end
         end
   end
   CS.CurrentVariables[x] = 1
   CS.CurrentObjectiveValue+=CS.Variables[x]
   return true,CS
end





#=
Storing this :
function IsConsistent(CS::CurrentSolution)
   for j in 1:1:CS.NBconstraints
         if CS.LeftMembers_Constraints[j][CS.CurrentVarIndex] == 1
            if CS.LastRightMemberValue_Constraint[j] ==  0
               CS.LastRightMemberValue_Constraint[j] = 1
            else
               return false,nothing
            end
         end
   end
   CS.CurrentObjectiveValue+=CS.Variables[CS.CurrentVarIndex]
   return true,CS
end

cs1 = CS
#On change la valeur "actuelle" du mbr de droite de la variable que l'on passe a zero
for j in 1:1:cs1.NBconstraints
      if cs1.LeftMembers_Constraints[j][x] == 1
         cs1.LastRightMemberValue_Constraint[j] = 0
      end
end
cs1.CurrentVarIndex=y
cs1.CurrentObjectiveValue-= cs1.Variables[x]
answer, cs1 = IsConsistent(cs1)

=#
