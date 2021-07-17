# Authors: Qianqian Wan, Robert Ralston, Layla Unger, Mengcun Gao, Olivera Savic, Nathaniel Blanco

##################################
########## DESCRIPTION ###########
##################################

# This script implements the ALCOVE model described in Kruschke 1992.
# See the paper for a full description of the model.

# When using this model, it is assumed that you have a real or fake dataset consisting
# of a sequence of trials. In each trial, a stimulus is presented to a learner. The learner is 
# prompted to choose a category label for the stimulus, and given corrective feedback.

# This script implements the process of using ALCOVE to model category learning from the
# same sequence of stimuli as in your dataset, and identifying parameter values that capture
# the estimate the responses of the real or simulated learner.

# Below is a description of aspects of the dataset and model parameters

### CATEGORY LEARNING DATASET ###

# A matrix of stimuli in which each row is an exemplar, each column is a dimension

# A matrix of feedback: i.e., true category labels in which each row is an exemplar. 
# Labels are specified using one-hot encoding, such that the number of columns = number of
# categories, and each category corresponds to a 1 in one column (0s in others)

# A matrix of participant responses in the same format as the category labels matrix

########## PARAMETERS ##########

# Free parameters estimated from data

# c: specificity. The narrowness/width of the similarity kernel used when comparing 
#    similarity of current exemplar to hidden nodes. Larger values = narrower kernels,
#    such that similarity between exemplar and a hidden node drops off rapidly with distance
#    between them.
# lambdaA: learning rate for attention strengths between dimensions and hidden nodes 
# lambdaW: learning rate for association weights between hidden nodes and category output nodes
# phi: real-valued mapping constant that governs response determinacy. higher values = greater
#      likelihood of choosing most activated category output node as the response

# Fixed parameters
# r=1, constant determining the psychological distance metric (1 = city block, 2 = euclidean)
# q=1, for exponential similarity gradient 

# Learned parameters 
# alpha[dimensions]: attention strength for each dimension that modifies distance/similarity on a dimension 
#        between an exemplar and the hidden nodes. A given dimension has the same alpha for all hidden nodes.
# W [hidden nodes, categories]: association weights between hidden nodes and category outputs 

####################################
######### FUNCTIONS: ALCOVE ########
####################################

### PART I: STIMULUS -> RESPONSE ###

# Equation 1
# The calculate_distance function is used as part of Equation 1. 
# It calculates the distance across all dimensions between the current exemplar stimulus,
# and a single hidden node. Distance is modulated by alpha (the attention strength of each
# dimension) and r (the fixed parameter for the distance metric)
calculate_distance <- function(exemplar, hidden, alpha, r=1) {
  # Initialize a variable that will the the total distance between the exemplar and hidden node across
  # all dimensions
  total_distance <- 0
  # For each dimension...
  for (i in 1:length(exemplar) ) {
    # Calculate the distance between the exemplar and hidden node, modified by attention strength and r
    dim_distance <- alpha[i]*(abs(exemplar[i] - hidden[i])**r)
    # Add to total_distance
    total_distance <- total_distance + dim_distance
  }
  final_distance <- total_distance**(1/r)
  return(final_distance)
}

# The activation_hidden function implements Equation 1.
# It calculates the activation of all hidden nodes given the current exemplar stimulus.
# It takes as input the current exemplar stimulus (represented as a vector of dimension values), 
# all hidden nodes (represented as a [hiddennode, dimensions] matrix), alphas, the free 
# parameter c, and the fixed parameters r and q. Alphas, c, r and q all modulate the similarity 
# between the current exemplar stimulus and the hidden nodes.
activation_hidden <- function(exemplar, hidden, alpha, c, r=1, q=1) {
  # Initialize a vector that will contain the distance between the exemplar and each hidden node
  distance <- rep(NA,nrow(hidden))
  # For each hidden node...
  for (node in 1:nrow(hidden)){
    # Calculate the distance between the exeplar and the hidden node
    distance[node] <- calculate_distance(exemplar, hidden[node,], alpha, r)
  }
  # Calculate the activation of each hidden node based on distance, c, and type of similarity gradient
  a_hidden <- exp( -c * (distance**q) )
  return(a_hidden)
}

# Equation 2
# The activation_output function calculates the activation of each category output node
# based on the activation of the hidden nodes, and W, the association weights between
# hidden nodes and output categories.  
# W is represented as a matrix: [hidden nodes, categories]
activation_output <- function(W, a_hidden){
  a_output <- a_hidden %*% W 
  return(a_output)
}

# Equation 3
# The probability_output funciton calculates the response probabilities for the categories.
# This function takes as input the activation of each category output node, and phi, the
# free parameter for response determinacy. 
probability_output <- function(phi, a_output){
  p_output <- exp(phi*a_output)/sum(exp(phi*a_output))
  return(p_output)
}

##### PART I: TEST RUN #####
# Initialize an ALCOVE model for stimuli with two dimensions and
# two hidden nodes (with dimension values 0,1 and 1,0), where one 
# hidden node is associted with category 1 (1, 0) and the other with
# category 2 (0, 1)
AlcoveList <- list(c = 1,   
                   lambdaW = 0.9,
                   lambdaA = 0.9,
                   phi = 10,
                   W = matrix(c(1,0,0,1),2,2, byrow=T), #[hiddennodes, categories]
                   alpha = c(0.5,0.5),
                   hiddennodes = matrix(c(1,0,0,1),2,2, byrow = T), #[hiddennodes, dimensions]
                   r = 1,
                   q = 1
)


exemplar1 = c(1,0) # Exemplar should be most similar to hiddennode 1, and therefore activate category 1
exemplar2 = c(0,1) # Exemplar should be most similar to hiddennode 2, and therefore activate category 2

test_hidden_activation <- activation_hidden(exemplar = exemplar1, hidden = AlcoveList$hiddennodes, 
                          alpha = AlcoveList$alpha, c = AlcoveList$c, 
                          r = AlcoveList$r, q = AlcoveList$q) 
test_hidden_activation

test_output_activation <- activation_output(W=AlcoveList$W, test_hidden_activation)
test_output_activation

probability_output(AlcoveList$phi, test_output_activation)

### PART I: FORWARD PASS ###

# The forward function combines all functions in Part I in which an exemplar
# stimulus leads to a set of category response probabilities
# It takes as input the current exemplar stimulus, and a list containing all 
# components of the ALCOVE model including free parameters, fixed parameters, 
# alphas, hidden nodes, and W. It returns the activation of the hidden and output
# nodes (used for updating in Part II), and the category response probabilities.
forward <- function(exemplar, AlcoveList){
  a_hidden <- activation_hidden(exemplar=exemplar, hidden=AlcoveList$hiddennodes,
                                   alpha = AlcoveList$alpha, c=AlcoveList$c,
                                   r=AlcoveList$r, q=AlcoveList$q)
  a_output <- activation_output(W=AlcoveList$W, a_hidden)
  p_output <- probability_output(AlcoveList$phi, a_output)
  
  forward_output <- list(activation_hidden = a_hidden, 
                         activation_output = a_output,
                         probability_output = p_output)
  return(forward_output)
}


forward_test <- forward(exemplar1, AlcoveList) 
forward_test


### PART II: FEEDBACK -> UPDATE ###

# Equation 4
# The humble_teacher function implements equation 4b. This modifies the feedback so that overprediction
# of correct category is not punished.
# Recall that feedback is a one-hot vector for the true category label (1 for category label, 0 for all other
# labels). Model prediction will be a vector of probabilities, with the highest value for the most predicted category.
# The function takes as input the activation of the category output nodes,
# and the feedback, which is a vector containing a 1 for the correct category and 0s 
# for the others (this will be taken from a row in the matrix for feedback across trials)
humble_teacher <- function(activation_output, feedback){
  teacher <- rep(NA,length(feedback))
  for (K in 1:length(feedback)){
    if (feedback[K] == 1){
      teacher[K] <- max(c(1,activation_output[K]))
    }else {
      teacher[K] <- min(c(-1,activation_output[K]))
    }
  }
  return(teacher)
}

# The error function implements equation 4a. This calculates the error in 
# terms of the difference between the humble teacher feedback and the activation
# of the category output nodes.
error <- function(teacher, activation_output){
  e <- 0.5*sum((teacher - activation_output)**2)
  return(e)
}


# Equation 5
# The deltaW function calculates the change in W based on error
deltaW <- function(lambdaW, teacher, activation_output, activation_hidden){
  activation_hidden <- matrix(activation_hidden, ncol = 1) # turn activation_hidden into a row matrix
  output_error <- matrix(teacher - activation_output, nrow = 1) # column matrix of errors: [1, categories]
  W_deltas <- lambdaW * (activation_hidden %*% output_error) # matrix of deltas for W[hidden nodes, categories] 
  return(W_deltas) 
}

# Equation 6
# The deltaAlpha function calculates the change in alpha based on error
deltaAlpha <- function(lambdaA, teacher, activation_output, activation_hidden, W, hidden, c, exemplar){
  
  output_error <- matrix(teacher - activation_output, nrow = 1) # column matrix of errors: [1 , categories] 
  W_transpose <- t(W) #Transpose W so that it is [categories, hidden nodes]
  weighted_errors <- output_error %*% W_transpose # [1, hidden nodes]
  
  alpha_deltas = rep(0,length(exemplar))
  
  for (node in 1: nrow(hidden)){
    alpha_deltas = alpha_deltas + weighted_errors[1, node] * activation_hidden[node] *
      c * abs(hidden[node,] - exemplar)
  }
  alpha_deltas = -lambdaA * alpha_deltas
  return(alpha_deltas)
}

# Test detlaW and deltaAlpha functions
deltaW (lambdaW = AlcoveList$lambdaW, teacher = c(1,-1),
        activation_output = forward_test$activation_output, 
        activation_hidden = forward_test$activation_hidden)

deltaAlpha (lambdaA = AlcoveList$lambdaA, teacher = c(1,-1), activation_output = forward_test$activation_output,
            activation_hidden = forward_test$activation_hidden, W = AlcoveList$W, 
            hidden = AlcoveList$hiddennodes, c = AlcoveList$c, exemplar = exemplar1)



### PART II: BACKWARD PASS ###

# The backward function combines all functions in Part II in which feedback
# updates the alphas and Ws
# It takes as input the current exemplar stimulus, a list containing all 
# components of the ALCOVE model including free parameters, fixed parameters, 
# alphas, hidden nodes, and W, the feedback containing the correct category label, 
# and the output of the forward function. 
# It returns an ALCOVE model in which alphas and Ws have been updated, and 
# the error for the trial.
backward <- function(exemplar, AlcoveList, label, forward_output){
  trial_teacher <- humble_teacher(forward_output$activation_output, label)
  
  trial_deltaW <- deltaW (lambdaW = AlcoveList$lambdaW, teacher = trial_teacher,
          activation_output = forward_output$activation_output, 
          activation_hidden = forward_output$activation_hidden)

  trial_deltaA <- deltaAlpha (lambdaA = AlcoveList$lambdaA, teacher = trial_teacher,
              activation_output = forward_output$activation_output,
              activation_hidden = forward_output$activation_hidden, 
              W = AlcoveList$W, hidden = AlcoveList$hiddennodes, 
              c = AlcoveList$c, exemplar = exemplar)
  
  AlcoveList$W = AlcoveList$W + trial_deltaW 
  AlcoveList$alpha = AlcoveList$alpha + trial_deltaA
  
  trial_error <- error(trial_teacher, forward_output$activation_output)
  backward_list = list(AlcoveList = AlcoveList, trial_error = trial_error)
  return(backward_list)
}

##### PART II: TEST RUN #####

# Matrix of exemplars across trials in which each row is a trial, each column is an dimension ([exemplars,dimensions])
# Generate matrix so that dimensions are binary, and each value occurs equally often
dimension_value <- rep(c(0,1),5)
exemplars <- matrix(c(sample(dimension_value, 10),sample(dimension_value, 10),sample(dimension_value, 10)),
                    nrow = 10, ncol = 3)
exemplars

# Matrix of correct category labels in which each row is a trial, each column is a category
# In this example, the first dimension of the exemplars perfectly determines category
labels <- cbind (exemplars[,1], 1-exemplars[,1])
labels 


AlcoveList <- list(c = 1,   
                   lambdaW = 0.9,
                   lambdaA = 0.9,
                   phi = 10,
                   W = matrix(c(1,0,0,1), nrow=2, ncol=2, byrow=T),
                   alpha = c(0.33, 0.33, 0.33),
                   hiddennodes = matrix(c(1,0,0,0,1,1), nrow=2, ncol=3, byrow = T),
                   r = 1,
                   q = 1
)

forward_test <- forward(exemplars[1,], AlcoveList) 
forward_test

backward(exemplars[1,], AlcoveList, label=labels[1,], forward_output = forward_test)

######################################
## FUNCTIONS: TRAINING & LIKELIHOOD ##
######################################

# The training function takes a set of free parameter values, training exemplars,
# and feedback, and returns a set of simulated category response probabilities for each
# trial. These will be used below in the likelihood function. 
# This function additionally returns errors for each trial, and the final version of the
# ALCOVE model at the end of training. In addition, you can return the ALCOVE model on each
# trial using the storeAlcove argument
training <- function(exemplars, AlcoveList, labels, storeAlcove = F){
  output_error = rep(NA, nrow(exemplars))
  simulated_response = matrix(NA, nrow = nrow(labels), ncol = ncol(labels))
  
  if (storeAlcove){
    output_AlcoveLists = rep(list(NA), nrow(exemplars))
  } 
  
  for (i in 1:nrow(exemplars)){
    forward_trial = forward(exemplars[i,], AlcoveList)
    backward_trial = backward(exemplars[i,], AlcoveList, labels[i,], forward_output = forward_trial)
    output_error[i] = backward_trial$trial_error
    simulated_response[i,] = forward_trial$probability_output
    if (storeAlcove){
      output_AlcoveLists[[i]] = backward_trial$AlcoveList
    }
    AlcoveList = backward_trial$AlcoveList
  }
  
  if (storeAlcove){
    return(list(output_AlcoveLists = output_AlcoveLists, 
                output_error = output_error, response_probabilities = simulated_response))
  }else{
    return(list(output_AlcoveList = AlcoveList, output_error = output_error, 
                response_probabilities = simulated_response))
  }
}

# Test training function with option to store updated ALCOVE model after each trial set to TRUE
training_test <- training(exemplars = exemplars, AlcoveList = AlcoveList, labels = labels, storeAlcove = T) 
training_test


# The initialize_alcove function takes a set of parameters and an argument that specifies
# whether to initialize hidden nodes as either: (1) Nodes with the same dimension values as the exemplars
# seen during category learning, or (2) Nodes that are evenly spaced over the dimensions (i.e., the "covering map")
initialize_alcove <- function(param, exemplars, labels, covering = FALSE, density = NULL, 
                              lowerbound = NULL, upperbound =NULL){
  no_categories <- ncol(labels)
  no_dimensions <- ncol(exemplars)
  if (covering){
    dimension_pos <- rep(list(NA),no_dimensions)
    for (i in 1:length(dimension_pos)){
      dimension_pos[[i]] <- seq(lowerbound[i],upperbound[i], length.out = density+2) 
    }
    hiddennodes <- as.matrix(expand.grid(dimension_pos))
  }else{
    hiddennodes <- exemplars  
  }
  AlcoveList <- list(c = param[1], #similarity gradient 30  
                     lambdaW = param[2],
                     lambdaA = param[3],
                     phi = param[4],
                     
                     alpha = rep(0.5, no_dimensions),
                     W = matrix(0, nrow(hiddennodes), no_categories),
                     hiddennodes = hiddennodes,
                     r = 1,
                     q = 1
  )
  return(AlcoveList)
}


# The likelihood function uses the training function to take a set of values for the free parameters,
# and return category response probabilities across trials. These response probabilities are compared to
# observed responses to estimate the likelihood for the parameter values.
# density = the number of points inbetween the bounds.
likelihood <- function(param, exemplars, labels, response, covering = FALSE, density = NULL, 
                       lowerbound = NULL, upperbound =NULL){
  
  AlcoveList <- initialize_alcove(param, exemplars, labels, covering = covering, density = density, 
                                  lowerbound = lowerbound, upperbound =upperbound)
  
  #training function for response probability 
  training_output <- training(exemplars, AlcoveList, labels, F) 
  resp_prob <- training_output$response_probabilities
  nlLikelihood = 0
  
  for (i in 1:nrow(response)){
    answer_index = which(response[i,]==1)
    nlLikelihood = nlLikelihood - log(resp_prob[i,answer_index])
  }
  return(nlLikelihood)
}

# Test likelihood with fake participant responses (all accurate for simplicity)
test_responses <- labels
likelihood(param = c(1,1,1,1), exemplars, labels, response = test_responses,
           covering = T, density = 0, 
           lowerbound = c(0,0,0), upperbound =c(1,1,1))

####################################
####### ESTIMATE PARAMETERS ########
####################################

# Likelihood function to minimize likelihood with optim
# Imposes upper limits on parameter values specified in a vector
# Upper limits are needed because extreme values produced during optimization
# can produce errors and are not interpretable

nllh_optim <- function(param, exemplars, labels, response, upperbound = c(30, 5, 5, 30)){
  new_param <- rep(NA, length(param))
  for(i in 1:length(param)){
      new_param[i] <- upperbound[i]/(1+exp(-param[i]))
  }
  
  likelihood_output <- likelihood(param=new_param, exemplars, labels, response)
  return(likelihood_output)
}

# Function that imposes the same upper bounds on the parameter values output by optim
# and returns parameter values as a dataframe
transform_output <- function(output_param, upperbound = c(30, 5, 5, 30)){
  new_param <- rep(NA, length(output_param))
  for(i in 1:length(output_param)){
      new_param[i] <- upperbound[i]/(1+exp(-output_param[i]))
  }
  new_param <- round(new_param, 2)
  c <- new_param[1]
  lambdaW <- new_param[2]
  lambdaA <- new_param[3]
  phi <- new_param[4]
  
  param <- data.frame(c, lambdaW, lambdaA, phi)
  
  return(param)
}

optim_output <- optim(c(20,1,1,.5), nllh_optim, exemplars = exemplars, labels = labels, response = test_responses)

transform_output(optim_output$par)