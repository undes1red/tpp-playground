import itertools, math

def extract_single_multiple_arguments_from_the_list(hyperparameter_list):
    single_parameters = {}
    multiple_parameters = {}

    last_parameter = ''
    for items in hyperparameter_list:
        if last_parameter == '':
            '''
            new arguments:
            '''
            last_parameter = items
        elif last_parameter.startswith('--') and type(items) == list:
            '''
            arguments with multiple choices
            '''
            multiple_parameters[last_parameter] = items
            last_parameter = ''
        elif last_parameter.startswith('--') and items.startswith('--'):
            '''
            store_true arguments
            '''
            single_parameters[last_parameter] = ''
            last_parameter = items
        elif last_parameter.startswith('--') and not items.startswith('--'):
            '''
            arguments with single choice
            '''
            single_parameters[last_parameter] = items
            last_parameter = ''
    if last_parameter != '':
        single_parameters[last_parameter] = ''

    return single_parameters, multiple_parameters


def task_index_generator(hyperparameter_list):
    if hyperparameter_list is None:
        return [[''],]

    # head = os.path.join(root_path, hyperparameter_list[0])
    single_parameters, multiple_parameters = extract_single_multiple_arguments_from_the_list(hyperparameter_list)

    # Now, map all fixed argument into a list.
    # fixed_arguments_part = [head] + [opt.procedure_name + '_' + opt.script_type] \
    fixed_arguments_part = list(itertools.chain.from_iterable(single_parameters.items()))
    
    count_of_each_multiple_hp = len(multiple_parameters)

    if count_of_each_multiple_hp == 0:
        '''
        No multi_hps, just return the fixed parameter set.
        '''
        return [fixed_arguments_part]
    
    '''
    Affirm that all hyperparameter lists have the same length.
    '''
    number_of_parameters = len(list(multiple_parameters.values())[0])
    for hyperparameter in multiple_parameters.values():
        assert len(hyperparameter) == number_of_parameters, "Index mode requires all parameter list must have the same length!"

    # set iterators, the first iterator is always the single directed iterator. We use it to decide when we quit the argument
    # generation loop.
    final_hyperparameter_list = []
    for index in range(number_of_parameters):
        choosed_value = {key: item[index] for (key, item) in multiple_parameters.items()}
        choosed_value_to_list = list(itertools.chain.from_iterable(choosed_value.items()))
        final_list = fixed_arguments_part + choosed_value_to_list
        final_hyperparameter_list.append(final_list)
        
    return final_hyperparameter_list
        

def task_counting_generator(hyperparameter_list):
    if hyperparameter_list is None:
        return [[''],]

    # head = os.path.join(root_path, hyperparameter_list[0])
    single_parameters, multiple_parameters = extract_single_multiple_arguments_from_the_list(hyperparameter_list)
    
    # Now, map all fixed argument into a list.
    # fixed_arguments_part = [head] + [opt.procedure_name + '_' + opt.script_type] \
    fixed_arguments_part = list(itertools.chain.from_iterable(single_parameters.items()))

    # set iterators, the first iterator is always the single directed iterator. We use it to decide when we quit the argument
    # generation loop.
    multi_hp_count = len(multiple_parameters.values())
    count_of_each_multiple_hp = [len(item) for item in multiple_parameters.values()]
    current_index_of_each_list = [0] * multi_hp_count
    the_number_of_task = math.prod(count_of_each_multiple_hp)

    if count_of_each_multiple_hp == []:
        # No multiple hp is present.
        return [fixed_arguments_part]
    else:
        final_hyperparameter_list = []
        for _ in range(the_number_of_task):
            choosed_value = {key: item[index] for (key, item), index in zip(multiple_parameters.items(), current_index_of_each_list)}
            choosed_value_to_list = list(itertools.chain.from_iterable(choosed_value.items()))
            final_list = fixed_arguments_part + choosed_value_to_list
            final_hyperparameter_list.append(final_list)

            current_index_of_each_list[-1] += 1
            add_mark = False
            for idx, (current_index, max_unreachable_index) in enumerate(zip(current_index_of_each_list[::-1], count_of_each_multiple_hp[::-1])):
                if add_mark:
                    current_index_of_each_list[multi_hp_count - idx - 1] += 1
                    if current_index_of_each_list[multi_hp_count - idx - 1]  >= max_unreachable_index:
                        current_index_of_each_list[multi_hp_count - idx - 1] = 0
                        add_mark = True
                    else:
                        add_mark = False
                if current_index >= max_unreachable_index:
                    current_index_of_each_list[multi_hp_count - idx - 1] = 0
                    add_mark = True
        
        return final_hyperparameter_list


def remove_empty_str(x):
    try:
        while 1:
            x.remove('')
    except:
        return x