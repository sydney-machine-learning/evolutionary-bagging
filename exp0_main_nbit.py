def prepare_nbit(dataset_name):
    class Object(object):
        pass
    if dataset_name == '6bit':
        n_bit = 6
        data_dict = dict()
        while len(data_dict) < 2**n_bit:
            new_seq = np.random.randint(0, 2, n_bit)
            s = ''.join([str(bit) for bit in new_seq])
            if s not in data_dict.keys():
                if np.unique(new_seq, return_counts=True)[1][0]%2 == 0:
                    data_dict[s] = 1
                else:
                    data_dict[s] = 0
        data = Object()
        data.data = []
        data.target = []
        for bits, label in data_dict.items():
            bits = [int(bit) for bit in bits]
            data.data.append(bits)
            data.target.append(label)
        data.data = np.asarray(data.data)
        data.target = np.asarray(data.target)
    elif dataset_name == '8bit':
        n_bit = 8
        data_dict = dict()
        while len(data_dict) < 2**n_bit:
            new_seq = np.random.randint(0, 2, n_bit)
            s = ''.join([str(bit) for bit in new_seq])
            if s not in data_dict.keys():
                if np.unique(new_seq, return_counts=True)[1][0]%2 == 0:
                    data_dict[s] = 1
                else:
                    data_dict[s] = 0
        data = Object()
        data.data = []
        data.target = []
        for bits, label in data_dict.items():
            bits = [int(bit) for bit in bits]
            data.data.append(bits)
            data.target.append(label)
        data.data = np.asarray(data.data)
        data.target = np.asarray(data.target)
    X = pd.DataFrame(data.data)
    y = pd.DataFrame(data.target)
    return X, y

def test_algo_nbit(dataset_name, 
                  n_exp,
                  metric,
                  n_biomes, 
                  n_iter,
                  n_select,
                  n_new_biomes,
                  tournament_size,
                  n_mutation,
                  mutation_rate,
                  size_coef, 
                  cv=True):
    X_train, y_train = prepare_nbit(dataset_name)
    # random forest
    train_rf_metrics = []
    for i in range(100):
        clf = RandomForestClassifier(n_estimators=n_biomes) 
        clf.fit(X_train, y_train.values.ravel())
        train_preds = clf.predict(X_train)
        train_rf_metrics.append(eval(f"{metric}_score(y_train, train_preds)"))
    # print('Random forest train metric:       ', round(sum(train_rf_metrics), 2))
    # print(np.std(train_rf_metrics))

    # bagging performance
    train_bagging_scores = []
    for i in range(100):
        clf = BaggingClassifier(n_estimators=n_biomes)     
        clf.fit(X_train, y_train.values.ravel())
        train_preds = clf.predict(X_train)
        train_bagging_scores.append(eval(f"{metric}_score(y_train, train_preds)"))
    # print('Bagging train metric:             ', round(sum(train_bagging_scores), 2))
    # print(np.std(train_bagging_scores))
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_biomes - n_select - n_new_biomes
    max_depth = X_train.shape[1]
    all_voting_train = []
    for t in range(n_exp):
        # init random bags of random sizes
        biomes = {}
        for i in range(n_biomes):
            biomes[i] = gen_new_biome(X_train, y_train, max_initial_size)
        # evaluate
        payoff_df = pd.DataFrame(columns=['bag' + str(i) for i in range(n_biomes)])
        evaluate_biomes(biomes, size_coef, metric, cv)
        cover_idx = set()
        for biome in biomes.values():
            cover_idx = cover_idx.union(set(biome['X'].index))
        cover = len(cover_idx)/len(X_train)
        payoff_df.loc[0, :] = [round(biomes[j]['payoff']*100, 1) for j in range(n_biomes)]
        voting_train = []
        for i in tqdm(range(n_iter)):
            # selection
            new_biomes, selected_ids = naive_selection(biomes, n_select)
            # add new biomes
            for j in range(n_new_biomes):
                if cover > 0.9:
                    new_biome = gen_new_biome(X_train, y_train, max_initial_size)
                    new_biome_idx = random.choice(list(set(range(n_biomes)) - set(new_biomes.keys())))
                    new_biomes[new_biome_idx] = new_biome
                else:
                    uncover_idx = list(set(X_train.index) - cover_idx)
                    if len(uncover_idx) > max_initial_size/2:
                        uncover_idx = random.sample(uncover_idx, k=int(max_initial_size/2))
                    new_biome = dict()
                    new_biome['X'] = X_train.loc[uncover_idx, :]
                    new_biome['y'] = y_train.loc[uncover_idx, :]
                    new_biome_idx = random.choice(list(set(range(n_biomes)) - set(new_biomes.keys())))
                    new_biomes[new_biome_idx] = new_biome
            # crossover
            _, crossover_pool_idx = naive_selection(biomes, n_crossover)
            random.shuffle(crossover_pool_idx)
            remaining_idx = list(set(range(n_biomes)) - set(new_biomes.keys()))
            random.shuffle(remaining_idx)
            for j in range(0, n_crossover, 2):
                parent1 = biomes[crossover_pool_idx[j]]
                parent2 = biomes[crossover_pool_idx[j + 1]]
                child1, child2 = crossover_with_instance_prob(parent1, 
                                                              parent2,
                                                              max_depth)
                new_biomes[remaining_idx[j]] = child1
                new_biomes[remaining_idx[j + 1]] = child2
            new_biomes, mutation_idx = mutation(new_biomes, X_train, y_train, n_mutation, mutation_size)
            cover_idx = set()
            for biome in new_biomes.values():
                cover_idx = cover_idx.union(set(biome['X'].index))
            cover = len(cover_idx)/len(X_train)
            # update population
            biomes = copy.deepcopy(new_biomes)
            # evaluate
            evaluate_biomes(biomes, size_coef, metric, cv)
            payoff_df.loc[i + 1, :] = [round(biomes[j]['payoff']*100, 1) for j in range(n_biomes)]
            voting_train.append(round(voting_metric(biomes, X_train, y_train, metric)*100, 2))
        best_iter = np.argmax(voting_train)
        all_voting_train.append(voting_train[best_iter])
    evo_acc = np.mean(all_voting_train)
    bag_acc = sum(train_bagging_scores)
    rf_acc = sum(train_rf_metrics)
    return evo_acc, bag_acc, rf_acc

def test_plot_nbit(dataset_name, 
                  start=10,
                  stop=100,
                  cv=True):
    evo = []
    bag = []
    rf = []
    for n in range(start, stop, 2):
        n_new_biomes = int(n*0.2)
        if n_new_biomes%2 != 0:
            n_new_biomes += 1
        tournament_size = 2
        n_mutation = int(n/10)
        mutation_size = 0
        size_coef = 100 if dataset_name=='6bit' else 1000
        evo_acc, bag_acc, rf_acc = test_algo_nbit(dataset_name, 
                                                  10,
                                                  'accuracy',
                                                  n, 
                                                  5,
                                                  0,
                                                  n_new_biomes,
                                                  tournament_size,
                                                  n_mutation,
                                                  mutation_size,
                                                  size_coef, 
                                                  cv)
        evo.append(evo_acc)
        bag.append(bag_acc)
        rf.append(rf_acc)
    print_df = pd.DataFrame({'EvoBagging': evo, 'Bagging': bag, 'Random Forest': rf})
    print_df.index = range(start, stop, 2)
    p = sns.lineplot(data=print_df)
    p.set_xlabel("Number of bags")
    p.set_ylabel("Accuracy")
    p.xaxis.set_major_locator(ticker.MultipleLocator(10))
    p.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig = p.get_figure()
    fig.savefig(f'viz/{dataset_name}.png')
    plt.clf()
    return print_df