library(stopwords)
library(tidyverse)
library(quanteda)
library(reticulate)
library(umap)
library(tidytext)
library(coop)

print("You will need a working installation of python with the following python packages installed: 'sentence-transformers' 'umap-learn' 'hdbscan'")

topic_model <- function(data, docs, doc_id, time_stamp, transformer_model, min_cluster_size, min_samples = NULL,
                        n = 10, stopwords = stopwords::data_stopwords_stopwordsiso$en,
                        split = F, avg_words = NULL, random = F, seed = 2117, seed_split = 1721) {

  if (split == 0 & random == 1) {stop("No random selection possible with unsplit documents")}
  if (split == 1 & is.null(avg_words) == 1) {stop("avg_words must be specified")}
  
  parameters <- list(min_cluster_size = min_cluster_size, min_samples = min_samples,
                     split = split, random = random, stopwords = stopwords)
  
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
                  set.seed(seed_split)
                  splits <- {if(random == 0) splits
                              else if (random == 1) splits %>% group_by(id) %>% slice_sample(n = 1)
                              else print("Error: random must be true/false")
                           }
                  splits
        } else {
               print("Error: split must be true/false")
        }
  
  
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
  umap_dimc$min_dist <- 0.0000000000000000000001
  umap_dimc$n_neighbors <- 30
  
  ##for plotting
  umap_dimv <- umap::umap.defaults
  umap_dimv$n_components <- 2
  umap_dimv$metric <- "cosine"
  umap_dimv$min_dist <- 0.0000000000000000000001
  umap_dimv$n_neighbors <- 30
  
  set.seed(seed)
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
  cluster <- tibble(labels = cluster$labels_,
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
  
  doc_per_topic <- docs_with_topic %>% 
                   group_by(topic) %>% 
                   summarize(doc = paste(doc, collapse = " "))
  
  c_tf_idf <- doc_per_topic %>% 
              tidytext::unnest_tokens(word, doc) %>% 
              filter(!word %in% stopwords) %>%
              count(topic, word, sort = T, name = "t") %>% 
              group_by(topic) %>% 
              mutate(w = sum(t)) %>% 
              mutate(tf = t / w) %>% 
              ungroup() %>% 
              add_count(word, name = "sum_t") %>% 
              mutate(idf = log(nrow(docs_with_topic) / sum_t)) %>% 
              mutate(tf_idf = tf * idf)
  
  topwords <- c_tf_idf %>% 
              group_by(topic) %>% 
              slice_max(tf_idf, n = n, with_ties = F) %>% 
              group_by(topic) %>% 
              arrange(desc(tf_idf), .by_group = T)
  
  topic_sizes <- docs_with_topic %>% 
                 group_by(topic) %>% 
                 count(name = "size") %>% 
                 arrange(desc(size))
  
  fin <- list(embedding = embed, reduced_embedding = umap, plotting_layout = plot,
              cluster = cluster, top_words = topwords, topic_sizes = topic_sizes,
              split_docs_with_topic = {if (split == 1 & random == 0) split_docs_with_topic},
              docs_with_topic = docs_with_topic, c_tf_idf = c_tf_idf,
              parameters = parameters, cluster_exemplars = cluster_exemplars)
  
  return(fin)
}

plot_topics <- function(topic_model, min_cluster_size = NULL, min_samples = NULL,
                        noise = F, split = T) {
  clusterer <- reticulate::import("hdbscan")
  umap <- topic_model[["reduced_embedding"]]
  plot <- topic_model[["plotting_layout"]]
  topic_names <- topic_model[["topic_names"]]
  split_docs_with_topic <- topic_model[["split_docs_with_topic"]]
  
  cluster <- if (is.null(min_cluster_size) == 1 & is.null(min_samples) == 1) {
                topic_model[["cluster"]]
                } else if (is.null(min_cluster_size) == 0 | is.null(min_samples) == 0) {
                          cluster <- clusterer$HDBSCAN(min_cluster_size = min_cluster_size,
                                                       min_samples = min_samples,
                                                       metric = "euclidean",
                                                       cluster_selection_method = "eom")$fit(umap$layout)
                          cluster_exemplars <- tibble(exemplars = cluster$exemplars_)
                          cluster <- tibble(labels = cluster$labels_,
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
      ggplot(aes(x = x, y = y, color = {if (is.null(topic_names) == 0) topic_name else as.character(topic)})) +
      geom_point() +
      theme_void() +
      theme(legend.position = "bottom") +
      labs(x = NULL, y = NULL, color = NULL)
  }
}

update_topics <- function(topic_model, min_cluster_size = NULL, min_samples = NULL, n = 10,
                          stopwords_add = "") {
  umap <- topic_model[["reduced_embedding"]]
  split_docs_with_topic <- topic_model[["split_docs_with_topic"]]
  split <- topic_model[["parameters"]][["split"]]
  random <- topic_model[["parameters"]][["random"]]
  stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add)
  min_cluster_size <- {if (is.null(min_cluster_size) == 1) topic_model[["parameters"]][["min_cluster_size"]]
                         else min_cluster_size}
  min_samples <- {if (is.null(min_samples) == 1) topic_model[["parameters"]][["min_samples"]]
                    else min_samples}
  
  #Topic moddeling
  ##clustering
  clusterer <- reticulate::import("hdbscan")
  
  cluster <- clusterer$HDBSCAN(min_cluster_size = min_cluster_size,
                               min_samples = min_samples,
                               metric = "euclidean",
                               cluster_selection_method = "eom")$fit(umap$layout)
  cluster_exemplars <- tibble(exemplars = cluster$exemplars_)
  cluster <- tibble(labels = cluster$labels_,
                    probabilities = cluster$probabilities_)

  split_docs_with_topic <- if (split == 0 | (split == 1 & random == 1)) {
                              split_docs_with_topic %>% 
                              ungroup() %>% 
                              mutate(topic = cluster$labels)
                           } else if (split == 1 & random == 0) {
                                     split_docs_with_topic %>% 
                                     ungroup() %>% 
                                     mutate(topic_para = cluster$labels,
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
  
  doc_per_topic <- docs_with_topic %>% 
                   group_by(topic) %>% 
                   summarize(doc = paste(doc, collapse = " "))
  
  c_tf_idf <- doc_per_topic %>% 
              tidytext::unnest_tokens(word, doc) %>% 
              filter(!word %in% stopwords) %>%
              count(topic, word, sort = T, name = "t") %>% 
              group_by(topic) %>% 
              mutate(w = sum(t)) %>% 
              mutate(tf = t / w) %>% 
              ungroup() %>% 
              add_count(word, name = "sum_t") %>% 
              mutate(idf = log(nrow(docs_with_topic) / sum_t)) %>% 
              mutate(tf_idf = tf * idf)
  
  topwords <- c_tf_idf %>% 
              group_by(topic) %>% 
              slice_max(tf_idf, n = n, with_ties = F) %>% 
              group_by(topic) %>% 
              arrange(desc(tf_idf), .by_group = T)
  
  topic_sizes <- docs_with_topic %>% 
                 group_by(topic) %>% 
                 count(name = "size") %>% 
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
  topic_model[["cluster_exemplars"]] <- cluster_exemplars
   
  return(topic_model)
}

topics_over_time <- function(topic_model, n = 5, timestamps = "docs_with_topic", stopwords_add = "") {
  docs_with_topic <- topic_model[[timestamps]]
  stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add)
  topic_names <- topic_model[["topic_names"]]
  
  over_time_l <- lapply(sort(unique(docs_with_topic$timestamp)), function(x) {
                   docs_at_time <- docs_with_topic %>% 
                                   filter(timestamp == x)
                   docs_at_time_per_topic <- docs_at_time %>% 
                                             group_by(topic) %>% 
                                             summarize(doc = paste(doc, collapse = " "))
                   c_tf_idf_at_time <- docs_at_time_per_topic %>% 
                                       tidytext::unnest_tokens(word, doc) %>% 
                                       filter(!word %in% stopwords) %>%
                                       count(topic, word, sort = T, name = "t") %>% 
                                       group_by(topic) %>% 
                                       mutate(w = sum(t)) %>% 
                                       mutate(tf = t / w) %>% 
                                       ungroup() %>% 
                                       add_count(word, name = "sum_t") %>% 
                                       mutate(idf = log(nrow(docs_at_time) / sum_t)) %>% 
                                       mutate(tf_idf = tf * idf)
                   topwords_at_time <- c_tf_idf_at_time %>% 
                                       group_by(topic) %>% 
                                       slice_max(tf_idf, n = n, with_ties = F) %>% 
                                       group_by(topic) %>% 
                                       arrange(desc(tf_idf), .by_group = T) %>% 
                                       summarize(words = paste(word, collapse = " ")) %>% 
                                       mutate(words = str_split(words, " ")) %>% 
                                       add_row(topic = setdiff(unique(docs_with_topic$topic), .$topic))
                   
                   topic_freq <- docs_at_time %>% 
                                 group_by(topic) %>% 
                                 count(topic, timestamp, name = "frequency") %>% 
                                 ungroup() %>% 
                                 mutate(percent = (frequency / sum(frequency)) * 100) %>% 
                                 add_row(topic = setdiff(unique(docs_with_topic$topic), .$topic),
                                         timestamp = unique(.$timestamp), frequency = 0, percent = 0)
                   
                   topics_over_time <- left_join(topwords_at_time, topic_freq, by = "topic")
                   return(topics_over_time)
  })
  
  over_time <- do.call(rbind, over_time_l) %>% 
               mutate(topic_name = factor(topic,
                                          levels = sort(unique(topic)),
                                          labels = c("Not assigned", topic_names)))
  return(over_time)
}

plot_timeline <- function(topics_over_time, percentage = T, scatter = F, noise = F, alpha = 1) {
  topics_over_time %>% 
    mutate(topic = as.character(topic)) %>% 
    {if (noise == 0) filter(., topic != "-1")
       else if (noise == 1) .
       else print("Error: noise must be true/false")
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
             select(topic, word, tf_idf) %>% 
             filter(topic != "-1") %>% 
             arrange(topic) %>% 
             pivot_wider(names_from = topic, values_from = tf_idf) %>% 
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

merge_topics_manual <- function(topic_model, from, into, n = 10, stopwords_add = NULL) {
  docs_with_topic <- topic_model[["docs_with_topic"]] %>% 
                     mutate(topic = replace(topic, topic == from, into))
  stopwords <- c(topic_model[["parameters"]][["stopwords"]], stopwords_add)
  
  doc_per_topic <- docs_with_topic %>%
                   group_by(topic) %>% 
                   summarize(doc = paste(doc, collapse = " "))
  
  c_tf_idf <- doc_per_topic %>% 
              tidytext::unnest_tokens(word, doc) %>% 
              filter(!word %in% stopwords) %>%
              count(topic, word, sort = T, name = "t") %>% 
              group_by(topic) %>% 
              mutate(w = sum(t)) %>% 
              mutate(tf = t / w) %>% 
              ungroup() %>% 
              add_count(word, name = "sum_t") %>% 
              mutate(idf = log(nrow(docs_with_topic) / sum_t)) %>% 
              mutate(tf_idf = tf * idf)
  
  topwords <- c_tf_idf %>% 
              group_by(topic) %>% 
              slice_max(tf_idf, n = n, with_ties = F) %>% 
              group_by(topic) %>% 
              arrange(desc(tf_idf), .by_group = T)
  
  topic_sizes <- docs_with_topic %>% 
                 group_by(topic) %>% 
                 count(name = "size") %>% 
                 arrange(desc(size))
  
  topic_model[["docs_with_topic"]] <- docs_with_topic
  topic_model[["c_tf_idf"]] <- c_tf_idf
  topic_model[["top_words"]] <- topwords
  topic_model[["topic_sizes"]] <- topic_sizes
   
  return(topic_model)
}

merge_topics <- function(topic_model, min_topic_size, n = 10, stopwords_add = NULL) {
  
  min <- min(topic_model[["topic_sizes"]]$size)
  
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
                 filter(topic == from)
    into <- into_temp$topic_comp[which.max(into_temp$cos_sim)] %>% 
            as.numeric()
    
    split_docs_with_topic <- split_docs_with_topic %>% 
                             mutate(topic_para = replace(topic_para, topic_para == from, into),
                                    topic = replace(topic, topic == from, into))
    
    docs_with_topic <- docs_with_topic %>% 
                       mutate(topic = replace(topic, topic == from, into))
    
    cluster <- cluster %>% 
               mutate(labels = replace(labels, labels == from, into))
    
    doc_per_topic <- docs_with_topic %>%
                     group_by(topic) %>% 
                     summarize(doc = paste(doc, collapse = " "))
    
    c_tf_idf <- doc_per_topic %>% 
                tidytext::unnest_tokens(word, doc) %>% 
                filter(!word %in% stopwords) %>%
                count(topic, word, sort = T, name = "t") %>% 
                group_by(topic) %>% 
                mutate(w = sum(t)) %>% 
                mutate(tf = t / w) %>% 
                ungroup() %>% 
                add_count(word, name = "sum_t") %>% 
                mutate(idf = log(nrow(docs_with_topic) / sum_t)) %>% 
                mutate(tf_idf = tf * idf)
    
    topwords <- c_tf_idf %>% 
                group_by(topic) %>% 
                slice_max(tf_idf, n = n, with_ties = F) %>% 
                group_by(topic) %>% 
                arrange(desc(tf_idf), .by_group = T)
    
    topic_sizes <- docs_with_topic %>% 
                   group_by(topic) %>% 
                   count(name = "size") %>% 
                   arrange(desc(size))
    
    min <- min(topic_sizes$size)
    
    topic_model[["cluster"]] <- cluster
    topic_model[["docs_with_topic"]] <- docs_with_topic
    topic_model[["split_docs_with_topic"]] <- split_docs_with_topic
    topic_model[["c_tf_idf"]] <- c_tf_idf
    topic_model[["top_words"]] <- topwords
    topic_model[["topic_sizes"]] <- topic_sizes
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
  split_docs_with_topic <- split_docs_with_topic %>% 
                           mutate(topic_name = recode(topic, !!!setNames(c("Not assigned", topic_names),
                                                                         sort(unique(split_docs_with_topic$topic)))))
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
