def strawberryfields_to_neuroptica_clements(tlist, N):
    #Rearrange tlist to be compatible with neuroptica
    tlist_straw = []
    tlist_idx = 0

    #Arrange tlist into list of diagonals
    #N-1 diagonals
    for i in range(N-1):
        diag_list = []
        #Each diagonal up to N/2 has 2*i+1 elements
        if i < N/2:
            for j in range(2*i+1):
                diag_list.append(tlist[tlist_idx])
                tlist_idx += 1
        #Each diagonal after N/2 has 2*((N-1)-i) elements 
        else:
            for j in range(2*((N-1)-i)):
                diag_list.append(tlist[tlist_idx])
                tlist_idx += 1
        tlist_straw.append(diag_list)

    #Convert diagonals to columns
    tlist_neur = []
    tlist_temp = tlist_straw.copy()

    #Take first t from each diagonal then remove previous column
    for i in range(N):
        #Index of entry within column, not index of column
        col_idx = 0
        #Length of column is different for even and odd columns
        if i % 2 == 0:
            col_len = N/2
        else:
            col_len = N/2 - 1

        #Index of diagonal within matrix
        diag_num = 0

        #Remove first entries of first col_len non-empty diagonals
        while col_idx < col_len:
            if tlist_temp[diag_num]:
                tlist_neur.append(tlist_temp[diag_num][0])
                tlist_temp[diag_num] = tlist_temp[diag_num][1:]
                col_idx += 1
            diag_num += 1
    return tlist_neur

def strawberryfields_to_neuroptica_reck(tlist, N):
    #Rearrange tlist to be compatible with neuroptica
    tlist_straw = []
    tlist_idx = 0

    #Arrange tlist into list of diagonals
    #N-1 diagonals
    for i in range(N-1):
        diag_list = []
        #Each diagonal has i+1 elements
        for j in range(i+1):
            diag_list.append(tlist[tlist_idx])
            tlist_idx += 1
        tlist_straw.append(diag_list)

    #Convert diagonals to columns
    tlist_neur = []
    tlist_temp = tlist_straw.copy()

    #Take first t from each diagonal then remove previous column
    for i in range(2*N-2):
        #Reverse indices to reverse column direction
        for j in reversed(range(min(i,len(tlist_temp)))):
            if tlist_temp[j]:
                tlist_neur.append(tlist_temp[j][0])
                tlist_temp[j] = tlist_temp[j][1:]

    return tlist_neur
