#include <iostream>
#include <vector>
#include <iomanip>

std::string lcs(std::string a, std::string b, int m, int n) {
    std::vector<std::vector<int>> tab(m + 1, std::vector<int>(n + 1));
    // ------ padding ------
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++)
            tab[i][j] = -1;
    }

    for (int i = 0; i <= m; i++)
        tab[i][0] = 0;
    for (int i = 0; i <= n; i++)
        tab[0][i] = 0;
    // ---------------------
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (b[j - 1] == a[i - 1]) { // table is augmented, so need to -1 in string
                // point to diagonal elem.
                tab[i][j] = tab[i - 1][j - 1] + 1;
            }
            else {
                // point to max of prev row/clmn elem
                tab[i][j] = std::max(tab[i - 1][j], tab[i][j - 1]);
            }
        }
    }

    for (int i = 0; i < tab.size(); i++) {
        for (int j = 0; j < tab[i].size(); j++)
            std::cout << tab[i][j] << std::setw(2);
        std::cout << "\n";
    }

    // table traversal from the very last elem

    std::string seq;
    int r_trav = m;
    int c_trav = n;
    while (r_trav != 0 && c_trav != 0) {
        if (tab[r_trav][c_trav] > tab[r_trav - 1][c_trav - 1]) {
            seq += a[r_trav - 1];
            r_trav -= 1;
            c_trav -= 1;
        }

        else if (tab[r_trav][c_trav] == tab[r_trav - 1][c_trav])
            r_trav -= 1;
        else if (tab[r_trav][c_trav] == tab[r_trav][c_trav - 1])
            c_trav -= 1;
    }
    std::reverse(seq.begin(), seq.end());
    return seq;
}