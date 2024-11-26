if (!require("pacman")) install.packages("pacman"); library(pacman)
pacman::p_load("stopwords", "tidyverse", "quanteda", "reticulate", "umap", "tidytext", "coop", "cleanNLP")

print(paste("You will need a working installation of python with the following python packages installed:",
            "'sentence-transformers' 'umap-learn' 'hdbscan'. If you want to use the lemmatised version of words",
            "to calculate topic representations you also need the python package 'spacy' for cleannlp to work"))

topic_model <- function(data,
                        docs,                                                   #variable that stores documents/text
                        doc_id,                                                 #variable with unique identifier per document
                        time_stamp,                                             #variable that stores time stamps (needed only for time series topic modelling)
                        transformer_model,                                      #sBERT transformer model from https://huggingface.co/models?library=sentence-transformers. recommendation: "sentence-transformers/all-mpnet-base-v2"
                        min_cluster_size,                                       #minimum number of documents for a topic (smaller means more topics). Defaults to 5% of document number
                        min_samples,                                            #defaults to min_cluster_size. When smaller fewer data points will be classified as noise
                        umap_dist = 0.00000000000000000001,                     #must be between 0 and 1. lower values allow data points to be closer together, meaning better clustering
                        umap_neighbours = 15,                                   #determines focus on local vs global structure in data. smaller means more topics
                        n_topwords = 10,                                        #number of words to show per topic. can easily be changed later
                        lemma = F,                                              #uses cleanNLP to lemmatise all words before calculating c-tf-idf for topic representations
                        spacy_model = "en_core_web_sm",                         #cleanNLP spacy model to use for lemmatisation, edit required for non-English data
                        stopwords = stopwords::data_stopwords_stopwordsiso$en,  #stopwords to exclude for c-tf-idf calculation
                        split = F,                                              #should documents be split into shorter documents? transformer models have input token limits
                        avg_words,                                              #splitting algorithm keeps sentences intact, thus average here and not absolute limit, some documents will be longer than this number. choose something here that fits the input token limit of the model you choose
                        random = F,                                             #if true randomly chooses one of the documents each document was split into. mainly works as a robustness check
                        seed = 2117,                                            #seed for consistent dimension reduction results. set to NULL to speed up computation, but results may vary across runs as a consequence
                        seed_split                                              #seed used for randomly choosing one part of each split document
                        ) {

  if (!is.data.frame(data)) {stop("data must be a data frame or tibble")}
  if (!is.logical(lemma)) {stop("lemma must be true/false")}
  if (!is.logical(split)) {stop("split must be true/false")}
  if (!is.logical(random)) {stop("random must be true/false")}
  
  if (split == 0 & random == 1) {stop("No random selection possible with unsplit documents")}
  if (split == 1 & missing(avg_words)) {stop("avg_words must be specified to split documents")}
  
  if (umap_dist == 0) {stop("'umap_dist' must be > 0")}
  
  if (missing(time_stamp)) {time_stamp = 0}
  
  if (missing(min_cluster_size)) {min_cluster_size <- (nrow(data)*0.05)}
  if (missing(min_samples)) {min_samples <- min_cluster_size}
  
  df <- if (split == 0) {
           data %>% 
           rename(documents = all_of(docs),
                  timestamp = all_of(time_stamp),
                  id = all_of(doc_id))
        } else if (split == 1) {
                  splits <- apply(data, 1, function(x) {
                              sen_toks_l <- quanteda::tokenize_sentence(x[[docs]])
                              sen_toks <- sen_toks_l[[1]]
                              sen_num <- length(sen_toks)
                              total_words <- ceiling(sum(lengths(strsplit(sen_toks, " "))))
                              num_chunks <- ceiling(total_words / avg_words)
                              chunk_size <- ceiling(sen_num / num_chunks)
                              chunk_sen <- split(sen_toks, ceiling(seq_along(sen_toks) / chunk_size))
                              paragraph <- sapply(chunk_sen, function(x) {
                                             paste0(x, collapse = " ")
                              })
                              out <- tibble(id = x[[doc_id]],
                                            documents = paragraph,
                                            timestamp = x[[time_stamp]])
                              out
                  })
                  splits <- do.call(rbind, splits)
                  {if (!missing(seed_split)) set.seed(seed_split)}
                  splits <- {if(random == 0) splits
                              else if (random == 1) splits %>% slice_sample(n = 1, .by = id)
                           }
                  splits
        }
  
  if (missing(min_cluster_size) & (split == 1 | random == 0)) {
    min_cluster_size <- (nrow(df)*0.05)
  }
  
  min_cluster_size <- as.integer(min_cluster_size)
  min_samples <- as.integer(min_samples)
  
  parameters <- list(transformer_model = transformer_model, min_cluster_size = min_cluster_size,
                     min_samples = min_samples, umap_dist = umap_dist,
                     umap_neighbours = umap_neighbours, lemma = lemma, spacy_model = spacy_model,
                     split = split, random = random, stopwords = stopwords)
  
  #document embeddings
  transformer <- reticulate::import("sentence_transformers")
  tokenizer <- transformer$SentenceTransformer(transformer_model)
  
  embed <- tokenizer$encode(df$documents,
                            show_progress_bar = T)
  
  #Dimension reduction
  ##for clustering
  umap_dimc <- umap::umap.defaults
  umap_dimc$n_components <- 10
  umap_dimc$metric <- "cosine"
  umap_dimc$min_dist <- umap_dist
  umap_dimc$n_neighbors <- umap_neighbours
  umap_dimc$random_state <- seed
  
  ##for plotting
  umap_dimv <- umap::umap.defaults
  umap_dimv$n_components <- 2
  umap_dimv$metric <- "cosine"
  umap_dimv$min_dist <- umap_dist
  umap_dimv$n_neighbors <- umap_neighbours
  umap_dimv$random_state <- seed
  
  umap <- umap::umap(embed,
                     method = "umap-learn",
                     config = umap_dimc)
  plot <- umap::umap(embed,
                     method = "umap-learn",
                     config = umap_dimv)
  
  #Topic moddeling
  ##clustering
  clusterer <- reticulate::import("hdbscan")
  
  cluster <- clusterer$HDBSCAN(min_cluster_size = min_cluster_size,
                               min_samples = min_samples,
                               metric = "euclidean",
                               cluster_selection_method = "eom")$fit(umap$layout)
  cluster_exemplars <- tibble(exemplars = cluster$exemplars_)
  cluster <- tibble(labels = as.numeric(as.character(cluster$labels_)),
                    probabilities = cluster$probabilities_)

  
  ##articles per cluster pasted together
  split_docs_with_topic <- if (split == 0 | (split == 1 & random == 1)) {
                              tibble(doc = df$documents,
                                     topic = cluster$labels,
                                     id = df$id,
                                     timestamp = df$timestamp)
                           } else if (split == 1 & random == 0) {
                                     tibble(id = df$id,
                                            timestamp = df$timestamp,
                                            doc = df$documents,
                                            topic_para = cluster$labels,
                                            prob = cluster$probabilities) %>% 
                                     group_by(id) %>% 
                                     mutate(topic = ifelse(n_distinct(topic_para) == 1,
                                                           topic_para,
                                                           ifelse(length(which(table(topic_para) == max(table(topic_para)))) == 1,
                                                                  ifelse(-1 %in% names(which.max(table(topic_para))),
                                                                         ifelse(length(which(table(topic_para) == max(table(topic_para)[-1]))) == 1,
                                                                                ifelse(-1 %in% names(table(topic_para)),
                                                                                       as.numeric(names(which.max(table(topic_para)[-1]))),
                                                                                       as.numeric(names(which.max(table(topic_para))))),
                                                                                topic_para[which.max(prob)]),
                                                                         ifelse(-1 %in% names(table(topic_para)),
                                                                                as.numeric(names(which.max(table(topic_para)[-1]))),
                                                                                as.numeric(names(which.max(table(topic_para)))))),
                                                                  topic_para[which.max(prob)])))
                           }
  
  docs_with_topic <- if (split == 0 | (split == 1 & random == 1)) {
                        split_docs_with_topic
                     } else if (split == 1 & random == 0) {
                               split_docs_with_topic %>%  
                               summarize(doc = paste(doc, collapse = " "),
                                         topic = unique(topic),
                                         timestamp = unique(timestamp))
                     }
  
  if (lemma == 1) {
     cleanNLP::cnlp_init_spacy(model_name = spacy_model)
     annotated_docs <- cleanNLP::cnlp_annotate(docs_with_topic,
                                               text_name = "doc",
                                               doc_name = "id")
     
     docs_with_topic <- annotated_docs$token %>% 
                        summarise(doc_lem = paste(lemma, collapse = " "),
                                  .by = doc_id) %>% 
                        left_join(docs_with_topic,
                                  .,
                                  by = join_by(id == doc_id))
     
     doc_per_topic <- docs_with_topic %>% 
                      summarise(doc = paste(doc, collapse = " "),
                                doc_lem = paste(doc_lem, collapse = " "),
                                .by = topic) %>% 
                      mutate(topic = as.double(topic))
  } else {
     doc_per_topic <- docs_with_topic %>% 
                      summarize(doc = paste(doc, collapse = " "),
                                .by = topic)
  }

  
  c_tf_idf <- doc_per_topic %>% 
              {if (lemma == 0) tidytext::unnest_tokens(., word, doc)
                 else if (lemma == 1) tidytext::unnest_tokens(., word, doc_lem)
              } %>% 
              filter(!word %in% stopwords) %>%
              count(topic, word, sort = T, name = "t") %>%                      #frequency of each word within each topic
              mutate(w = sum(t),                                                #total words per topic
                     tf = (t / w),                                              #L1 normalised frequency to account for differences in topic size
                     .by = topic) %>%
              mutate(f = sum(t),
                     .by = word) %>%                                            #total frequency of each word across all topics
              mutate(A = mean(unique(.$w)),                                     #average number of words per topic   
                     idf = log(1 + (A / f)),                   
                     c_tf_idf = (tf * idf))
  
  topwords <- c_tf_idf %>% 
              slice_max(c_tf_idf,
                        n = n_topwords,
                        with_ties = T,
                        by = topic) %>% 
              arrange(topic,
                      desc(c_tf_idf))
  
  topic_sizes <- docs_with_topic %>% 
                 count(topic,
                       name = "size") %>% 
                 arrange(desc(size))
  
  fin <- list(embedding = embed, reduced_embedding = umap, plotting_layout = plot,
              cluster = cluster, top_words = topwords, topic_sizes = topic_sizes,
              split_docs_with_topic = {if (split == 1 & random == 0) split_docs_with_topic},
              docs_with_topic = docs_with_topic, c_tf_idf = c_tf_idf,
              parameters = parameters, cluster_exemplars = cluster_exemplars)
  
  return(fin)
}

plot_topics <- function(topic_model,
                        min_cluster_size,
                        min_samples,
                        noise = F,
                        split = T) {
  clusterer <- reticulate::import("hdbscan")
  umap <- topic_model[["reduced_embedding"]]
  plot <- topic_model[["plotting_layout"]]
  topic_names <- topic_model[["topic_names"]]
  split_docs_with_topic <- topic_model[["split_docs_with_topic"]]
  
  cluster <- if (missing(min_cluster_size) & missing(min_samples)) {
                topic_model[["cluster"]]
                } else if (!missing(min_cluster_size) | !missing(min_samples)) {
                          min_cluster_size <- {if (missing(min_cluster_size)) topic_model[["parameters"]][["min_cluster_size"]]
                                                  else min_cluster_size}
                          min_samples <- {if (missing(min_samples)) topic_model[["parameters"]][["min_samples"]]
                                             else min_samples}
                          cluster <- clusterer$HDBSCAN(min_cluster_size = min_cluster_size,
                                                       min_samples = min_samples,
                                                       metric = "euclidean",
                                                       cluster_selection_method = "eom")$fit(umap$layout)
                          cluster_exemplars <- tibble(exemplars = cluster$exemplars_)
                          cluster <- tibble(labels = as.numeric(as.character(cluster$labels_)),
                                            probabilities = cluster$probabilities_)
                }
  
  if (split == T) {
    plot$layout %>% 
      as_tibble() %>% 
      mutate(topic = as.character(cluster$labels),
             topic_name = factor(topic, levels = sort(unique(as.numeric(topic))), labels = c("Not assigned", topic_names))) %>% 
      {if (noise == 0) filter(., topic != "-1")
         else if (noise == 1) .
         else print("Error: noise must be true/false")
      } %>% 
      ggplot(aes(x = V1, y = V2, color = {if (is.null(topic_names) == 0) topic_name else topic})) +
      geom_point() +
      theme_void() +
      theme(legend.position = "bottom") +
      labs(x = NULL, y = NULL, color = NULL)
  } else if (split == F) {
    split_docs_with_topic %>% 
      cbind(as_tibble(plot$layout)) %>% 
      summarize(topic = unique(topic),
                x = mean(V1),
                y = mean(V2)) %>% 
      mutate(topic_name = factor(topic, levels = sort(unique(as.numeric(topic))), labels = c("Not assigned", topic_names))) %>% 
      {if (noise == 0) filter(., topic != "-1")
         else if (noise == 1) .
         else print("Error: noise must be true/false")
      } %>% 
      ggplot(aes(x = x, y = y, color = {if (is.null(topic_names) == 0) topic_name else topic})) +
      geom_point() +
      theme_void() +
      theme(legend.position = "bottom") +
      labs(x = NULL, y = NULL, color = NULL)
  }
}

update_topics <- function(topic_model,
                          min_cluster_size,
                          min_samples,
                          umap_neighbours,
                          n_topwords = 10,
                          lemma,
                          spacy_model,
                          stopwords_add = "") {
  embed <- topic_model[["embedding"]]
  umap <- topic_model[["reduced_embedding"]]
  split_docs_with_topic <- topic_model[["split_docs_with_topic"]]
  docs_with_topic <- topic_model[["docs_with_topic"]]
  split <- topic_model[["parameters"]][["split"]]
  random <- topic_model[["parameters"]][["random"]]
  lemma <- {if (missing(lemma)) topic_model[["parameters"]][["lemma"]]
              else lemma}
  lemma_calc <- {if (topic_model[["parameters"]][["lemma"]] == 0 & lemma == 1) T
                   else F}
  umap_neighbours <- {if (missing(umap_neighbours)) topic_model[["parameters"]][["umap_neighbours"]]
                        else umap_neighbours}
  umap_calc <- {if (umap_neighbours == topic_model[["parameters"]][["umap_neighbours"]]) F
                  else T}
  if (!is.logical(lemma)) {stop("lemma must be true/false")}
  spacy_model <- {if (missing(spacy_model)) topic_model[["parameters"]][["spacy_model"]]
                    else spacy_model}
  stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add)
  min_cluster_size <- {if (missing(min_cluster_size)) topic_model[["parameters"]][["min_cluster_size"]]
                         else as.integer(min_cluster_size)}
  min_samples <- {if (missing(min_samples)) topic_model[["parameters"]][["min_samples"]]
                    else as.integer(min_samples)}
  
  if (umap_calc == 1) {
     #Dimension reduction
     ##for clustering
     umap_dimc <- umap::umap.defaults
     umap_dimc$n_components <- 10
     umap_dimc$metric <- "cosine"
     umap_dimc$min_dist <- umap_dist
     umap_dimc$n_neighbors <- umap_neighbours
     umap_dimc$random_state <- seed
    
     ##for plotting
     umap_dimv <- umap::umap.defaults
     umap_dimv$n_components <- 2
     umap_dimv$metric <- "cosine"
     umap_dimv$min_dist <- umap_dist
     umap_dimv$n_neighbors <- umap_neighbours
     umap_dimv$random_state <- seed
    
     umap <- umap::umap(embed,
                        method = "umap-learn",
                        config = umap_dimc)
     plot <- umap::umap(embed,
                        method = "umap-learn",
                        config = umap_dimv)
  }
  
  #Topic moddeling
  ##clustering
  clusterer <- reticulate::import("hdbscan")
  
  cluster <- clusterer$HDBSCAN(min_cluster_size = min_cluster_size,
                               min_samples = min_samples,
                               metric = "euclidean",
                               cluster_selection_method = "eom")$fit(umap$layout)
  cluster_exemplars <- tibble(exemplars = cluster$exemplars_)
  cluster <- tibble(labels = as.numeric(as.character(cluster$labels_)),
                    probabilities = cluster$probabilities_)

  split_docs_with_topic <- if (split == 0) {
                              docs_with_topic %>% 
                              ungroup() %>% 
                              mutate(topic = cluster$labels)
                           } else if (split == 1 & random == 1) {
                                     split_docs_with_topic %>% 
                                     ungroup() %>% 
                                     mutate(topic = cluster$labels)
                           } else if (split == 1 & random == 0) {
                                     split_docs_with_topic %>% 
                                     ungroup() %>% 
                                     mutate(topic_para = cluster$labels,
                                            prob = cluster$probabilities) %>%
                                     mutate(topic = ifelse(n_distinct(topic_para) == 1,
                                                           topic_para,
                                                           ifelse(length(which(table(topic_para) == max(table(topic_para)))) == 1,
                                                                  ifelse(-1 %in% names(which.max(table(topic_para))),
                                                                         ifelse(length(which(table(topic_para) == max(table(topic_para)[-1]))) == 1,
                                                                                ifelse(-1 %in% names(table(topic_para)),
                                                                                       as.numeric(names(which.max(table(topic_para)[-1]))),
                                                                                       as.numeric(names(which.max(table(topic_para))))),
                                                                                topic_para[which.max(prob)]),
                                                                         ifelse(-1 %in% names(table(topic_para)),
                                                                                as.numeric(names(which.max(table(topic_para)[-1]))),
                                                                                as.numeric(names(which.max(table(topic_para)))))),
                                                                  topic_para[which.max(prob)])),
                                            .by = id) %>% 
                                     mutate(topic = as.factor(topic))
                           }
  docs_with_topic <- if (split == 0 | (split == 1 & random == 1)) {
                        split_docs_with_topic
                     } else if (split == 1 & random == 0) {
                       split_docs_with_topic %>%  
                       summarize(doc = paste(doc, collapse = " "),
                                 topic = unique(topic),
                                 timestamp = unique(timestamp))
                     }
  
  if (lemma_calc == 1) {
     cleanNLP::cnlp_init_spacy(model_name = spacy_model)
     annotated_docs <- cleanNLP::cnlp_annotate(docs_with_topic,
                                               text_name = "doc",
                                               doc_name = "id")
     
     docs_with_topic <- annotated_docs$token %>% 
                        summarise(doc_lem = paste(lemma, collapse = " "),
                                  .by = doc_id) %>% 
                        left_join(docs_with_topic,
                                  .,
                                  by = join_by(id == doc_id))
     
     doc_per_topic <- docs_with_topic %>% 
                      summarise(doc = paste(doc, collapse = " "),
                                doc_lem = paste(doc_lem, collapse = " "),
                                .by = topic)
  } else if (lemma == 1 & lemma_calc == 0) {
     doc_per_topic <- docs_with_topic %>% 
                      summarise(doc = paste(doc, collapse = " "),
                                doc_lem = paste(doc_lem, collapse = " "),
                                .by = topic)
  } else if (lemma == 0) {
     doc_per_topic <- docs_with_topic %>% 
                      summarise(doc = paste(doc, collapse = " "),
                                .by = topic)
  }
  
  c_tf_idf <- doc_per_topic %>% 
              {if (lemma == 0) tidytext::unnest_tokens(., word, doc)
                else if (lemma == 1) tidytext::unnest_tokens(., word, doc_lem)} %>% 
              filter(!word %in% stopwords) %>%
              count(topic, word, sort = T, name = "t") %>%                      #frequency of each word within each topic
              mutate(w = sum(t),                                                #total words per topic
                     tf = (t / w),                                                #L1 normalised frequency to account for differences in topic size
                     .by = topic) %>%
              mutate(f = sum(t),
                     .by = word) %>%                                            #total frequency of each word across all topics
              mutate(A = mean(unique(.$w)),                                     #average number of words per topic   
                     idf = log(1 + (A / f)),                   
                     c_tf_idf = (tf * idf))
  
  topwords <- c_tf_idf %>% 
              slice_max(c_tf_idf,
                        n = n_topwords,
                        with_ties = T,
                        by = topic) %>% 
              arrange(topic,
                      desc(c_tf_idf))
  
  topic_sizes <- docs_with_topic %>% 
                 count(topic,
                       name = "size") %>% 
                 arrange(desc(size))
  
  topic_model[["cluster"]] <- cluster
  topic_model[["split_docs_with_topic"]] <- split_docs_with_topic
  topic_model[["docs_with_topic"]] <- docs_with_topic
  topic_model[["c_tf_idf"]] <- c_tf_idf
  topic_model[["top_words"]] <- topwords
  topic_model[["topic_sizes"]] <- topic_sizes
  topic_model[["parameters"]][["stopwords"]] <- stopwords
  topic_model[["parameters"]][["min_samples"]] <- min_samples
  topic_model[["parameters"]][["min_cluster_size"]] <- min_cluster_size
  topic_model[["parameters"]][["umap_neighbours"]] <- umap_neighbours
  topic_model[["parameters"]][["lemma"]] <- lemma
  topic_model[["parameters"]][["spacy_model"]] <- spacy_model
  topic_model[["cluster_exemplars"]] <- cluster_exemplars
  topic_model[["reduced_embedding"]] <- umap
  topic_model[["plotting_layout"]] <- plot
   
  return(topic_model)
}

topics_over_time <- function(topic_model,
                             n_topwords = 5,
                             stopwords_add = "") {
  docs_with_topic <- topic_model[["docs_with_topic"]]
  stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add)
  topic_names <- topic_model[["topic_names"]]
  
  over_time_l <- lapply(sort(unique(docs_with_topic$timestamp)), function(x) {
                   docs_at_time <- docs_with_topic %>% 
                                   filter(timestamp == x)
                   docs_at_time_per_topic <- docs_at_time %>%  
                                             summarize(doc = paste(doc, collapse = " "),
                                                       .by = topic) %>% 
                                             mutate(topic = as.double(topic))
                   c_tf_idf_at_time <- docs_at_time_per_topic %>% 
                                       {if (lemma == 0) tidytext::unnest_tokens(., word, doc)
                                         else if (lemma == 1) tidytext::unnest_tokens(., word, doc_lem)} %>% 
                                       filter(!word %in% stopwords) %>%
                                       count(topic, word, sort = T, name = "t") %>%                      #frequency of each word within each topic
                                       mutate(w = sum(t),                                                #total words per topic
                                              tf = (t / w),                                                #L1 normalised frequency to account for differences in topic size
                                              .by = topic) %>%
                                       mutate(f = sum(t),
                                              .by = word) %>%                                            #total frequency of each word across all topics
                                       mutate(A = mean(unique(.$w)),                                     #average number of words per topic   
                                              idf = log(1 + (A / f)),                   
                                              c_tf_idf = (tf * idf))
                   
                   topwords_at_time <- c_tf_idf_at_time %>%
                                       slice_max(c_tf_idf,
                                                 n = n_topwords,
                                                 with_ties = T,
                                                 by = topic) %>% 
                                       arrange(topic, desc(c_tf_idf)) %>% 
                                       summarize(words = paste(word, collapse = " ")) %>% 
                                       mutate(words = str_split(words, " ")) %>% 
                                       add_row(topic = setdiff(unique(docs_with_topic$topic), .$topic))
                   
                   topic_freq <- docs_at_time %>%
                                 count(topic, timestamp, name = "frequency") %>% 
                                 mutate(percent = (frequency / sum(frequency)) * 100) %>% 
                                 add_row(topic = setdiff(unique(docs_with_topic$topic), .$topic),
                                         timestamp = unique(.$timestamp), frequency = 0, percent = 0)
                   
                   topics_over_time <- left_join(topwords_at_time, topic_freq, by = "topic")
                   return(topics_over_time)
  })
  
  over_time <- bind_rows(over_time_l) %>% 
               mutate(topic_name = factor(topic,
                                          levels = sort(unique(topic)),
                                          labels = c("Not assigned", topic_names)))
  return(over_time)
}

plot_timeline <- function(topics_over_time,
                          percentage = T,
                          scatter = F,
                          noise = F,
                          alpha = 1) {
  topics_over_time %>% 
    mutate(topic = as.character(topic)) %>% 
    {if (noise == 0) filter(., topic != "-1")
       else .
    } %>% 
    ggplot(., aes(x = timestamp, y = {if (percentage == 1) percent else if (percentage == 0) frequency},
           group = topic_name, color = topic_name)) +
      {if (scatter == 1) geom_point(alpha = alpha) else if (scatter == 0) geom_line()} +
      theme_classic() +
      labs(y = ifelse(percentage == 1, "Percent", "Frequency"), color = "Topic") +
      scale_color_viridis_d(option = "mako", direction = -1, end = .8)
}


topic_simil <- function(topic_model, names = T) {
  c_tf_idf <- topic_model[["c_tf_idf"]]
  topic_names <- topic_model[["topic_names"]]
  
  cos_sim <- c_tf_idf %>% 
             select(topic, word, c_tf_idf) %>% 
             filter(topic != "-1") %>% 
             arrange(topic) %>% 
             pivot_wider(names_from = topic, values_from = c_tf_idf) %>% 
             column_to_rownames("word") %>% 
             mutate(across(everything(), ~ replace_na(., 0))) %>% 
             as.matrix() %>% 
             coop::cosine(., use = "complete.obs")
  
  if (names == 1) {
    rownames(cos_sim) <- topic_names
    colnames(cos_sim) <- topic_names
  } else if (names == 0) {
    rownames(cos_sim) <- unique(sort(c_tf_idf %>% filter(topic != -1) %>% pull(topic)))
    colnames(cos_sim) <- unique(sort(c_tf_idf %>% filter(topic != -1) %>% pull(topic)))
  }
  diag(cos_sim) <- NA
  
  return(cos_sim)
}

merge_topics_manual <- function(topic_model,
                                from,
                                into,
                                n_topwords = 10,
                                stopwords_add = NULL) {
  docs_with_topic <- topic_model[["docs_with_topic"]] %>% 
                     mutate(topic = replace(topic, topic == from, into))
  stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add)
  lemma <- topic_model[["parameters"]][["lemma"]]
  
  if (lemma == 1) {
       doc_per_topic <- docs_with_topic %>%
                        summarise(doc = paste(doc, collapse = " "),
                                  doc_lem = paste(doc_lem, collapse = " "),
                                  .by = topic)
    } else if (lemma == 0) {
       doc_per_topic <- docs_with_topic %>% 
                        summarise(doc = paste(doc, collapse = " "),
                                  .by = topic)
    }
  
  c_tf_idf <- doc_per_topic %>% 
              {if (lemma == 0) tidytext::unnest_tokens(., word, doc)
                 else if (lemma == 1) tidytext::unnest_tokens(., word, doc_lem)} %>% 
              filter(!word %in% stopwords) %>%
              count(topic, word, sort = T, name = "t") %>%                      #frequency of each word within each topic
              mutate(w = sum(t),                                                #total words per topic
                     tf = (t / w),                                                #L1 normalised frequency to account for differences in topic size
                     .by = topic) %>%
              mutate(f = sum(t),
                     .by = word) %>%                                            #total frequency of each word across all topics
              mutate(A = mean(unique(.$w)),                                     #average number of words per topic   
                     idf = log(1 + (A / f)),                   
                     c_tf_idf = (tf * idf))
  
  topwords <- c_tf_idf %>%
              slice_max(c_tf_idf,
                        n = n_topwords,
                        with_ties = T,
                        by = topic) %>% 
              arrange(topic, desc(c_tf_idf))
  
  topic_sizes <- docs_with_topic %>% 
                 count(topic, name = "size") %>% 
                 arrange(desc(size))
  
  topic_model[["docs_with_topic"]] <- docs_with_topic
  topic_model[["c_tf_idf"]] <- c_tf_idf
  topic_model[["top_words"]] <- topwords
  topic_model[["topic_sizes"]] <- topic_sizes
   
  return(topic_model)
}

merge_topics <- function(topic_model,
                         min_topic_size,
                         n_topwords = 10,
                         stopwords_add = NULL) {
  
  split <- topic_model[["parameters"]][["split"]]
  lemma <- topic_model[["parameters"]][["lemma"]]
  min <- min(topic_model[["topic_sizes"]]$size)
  merge <- tibble(from = numeric(),
                  into = numeric())
  while(min <= min_topic_size) {
    
    topic_sizes <- topic_model[["topic_sizes"]]
    split_docs_with_topic <- topic_model[["split_docs_with_topic"]]
    docs_with_topic <- topic_model[["docs_with_topic"]]
    cluster <- topic_model[["cluster"]]
    stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add) 
    
    from <- topic_sizes$topic[which.min(topic_sizes$size)]
    into_temp <- topic_simil(topic_model, names = F) %>% 
                 as.data.frame() %>% 
                 rownames_to_column(var = "topic") %>% 
                 pivot_longer(!starts_with("topic"), names_to = "topic_comp", values_to = "cos_sim") %>% 
                 filter(topic == from) %>% 
                 mutate(across(contains("topic"), as.numeric))
    into <- into_temp$topic_comp[which.max(into_temp$cos_sim)]
    
    if (split == 1) {
       split_docs_with_topic <- split_docs_with_topic %>% 
                                mutate(topic_para = replace(topic_para, topic_para == from, into),
                                       topic = replace(topic, topic == from, into))
    }
    
    docs_with_topic <- docs_with_topic %>% 
                       mutate(topic = replace(topic, topic == from, into))
    
    cluster <- cluster %>%
               mutate(labels = replace(labels, labels == from, into))
    
    if (lemma == 1) {
       doc_per_topic <- docs_with_topic %>%
                        summarise(doc = paste(doc, collapse = " "),
                                  doc_lem = paste(doc_lem, collapse = " "),
                                  .by = topic) %>% 
                        mutate(topic = as.double(topic))
    } else if (lemma == 0) {
       doc_per_topic <- docs_with_topic %>% 
                        summarise(doc = paste(doc, collapse = " "),
                                  .by = topic) %>% 
                        mutate(topic = as.double(topic))
    }
    
    c_tf_idf <- doc_per_topic %>% 
                {if (lemma == 0) tidytext::unnest_tokens(., word, doc)
                   else if (lemma == 1) tidytext::unnest_tokens(., word, doc_lem)} %>% 
                filter(!word %in% stopwords) %>%
                count(topic, word, sort = T, name = "t") %>%                      #frequency of each word within each topic
                mutate(w = sum(t),                                                #total words per topic
                       tf = (t / w),                                                #L1 normalised frequency to account for differences in topic size
                       .by = topic) %>%
                mutate(f = sum(t),
                       .by = word) %>%                                            #total frequency of each word across all topics
                mutate(A = mean(unique(.$w)),                                     #average number of words per topic   
                       idf = log(1 + (A / f)),                   
                       c_tf_idf = (tf * idf))
    
    topwords <- c_tf_idf %>% 
                slice_max(c_tf_idf,
                          n = n_topwords,
                          with_ties = T,
                          by = topic) %>% 
                arrange(topic, desc(c_tf_idf))
    
    topic_sizes <- docs_with_topic %>% 
                   count(topic, name = "size") %>% 
                   arrange(desc(size))
    
    min <- min(topic_sizes$size)
    
    merge <- merge %>% 
             add_row(from = from,
                     into = into)
    
    topic_model[["cluster"]] <- cluster
    topic_model[["docs_with_topic"]] <- docs_with_topic
    topic_model[["split_docs_with_topic"]] <- split_docs_with_topic
    topic_model[["c_tf_idf"]] <- c_tf_idf
    topic_model[["top_words"]] <- topwords
    topic_model[["topic_sizes"]] <- topic_sizes
    topic_model[["merge"]] <- merge
  }
  return(topic_model)
}

topic_names <- function(topic_model, topic_names) {
  top_words <- topic_model[["top_words"]]
  topic_sizes <- topic_model[["topic_sizes"]]
  docs_with_topic <- topic_model[["docs_with_topic"]]
  split_docs_with_topic <- topic_model[["split_docs_with_topic"]]
  
  top_words <- top_words %>% 
               mutate(topic_name = recode(topic, !!!setNames(c("Not assigned", topic_names),
                                                             sort(unique(top_words$topic)))))
  topic_sizes <- topic_sizes %>% 
                 mutate(topic_name = recode(topic, !!!setNames(c("Not assigned", topic_names),
                                                             sort(unique(topic_sizes$topic)))))
  try(split_docs_with_topic <- split_docs_with_topic %>% 
                               mutate(topic_name = recode(topic, !!!setNames(c("Not assigned", topic_names),
                                                                             sort(unique(split_docs_with_topic$topic))))), silent = T)
  docs_with_topic <- docs_with_topic %>% 
                     mutate(topic_name = recode(topic, !!!setNames(c("Not assigned", topic_names),
                                                                   sort(unique(docs_with_topic$topic)))))
  
  topic_model[["topic_names"]] <- topic_names
  topic_model[["top_words"]] <- top_words
  topic_model[["topic_sizes"]] <- topic_sizes
  topic_model[["split_docs_with_topic"]] <- split_docs_with_topic
  topic_model[["docs_with_topic"]] <- docs_with_topic
  return(topic_model)
}
