import numpy as np

from abstract_cipher import AbstractCipher



class Gift(AbstractCipher):

    def __init__(self, n_rounds=28, use_key_schedule=True):
        """
        Initializes a Present cipher object
        :param n_rounds: The number of round used for de-/encryption
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        """
        super(Gift, self).__init__(
            n_rounds,
            word_size=16, n_words=4, n_main_key_words=32, n_round_key_words=32, use_key_schedule=use_key_schedule
        )

    GIFT_RC = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
           0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
           0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
           0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
           0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
           0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28,
           0x10, 0x20];

    def get_n_rounds(self, key_additions=True):
        """
        :param key_additions: Whether to return the number of round key additions or the number of rounds
        :return: The number of rounds or the number of round key additions
        """
        if key_additions:
            return super(Gift, self).get_n_rounds()
        return super(Gift, self).get_n_rounds() + 1

    @staticmethod
    def bit_at_pos(X, M, n):
        """
        :param x: Array
        :param i: Bit position to return
        :return: The i-th bit of all elements in x
        """
        return ((X ^ ((X ^ (X >> n)) & M)) ^ (((X ^ (X >> n)) & M) << n))

    @staticmethod
    def bit_to_pos(A, B, M, n):
        """
        :param x: Array
        :param i: Bit position
        :return: All elements of x shifted to the left by i bits
        """
        return (A ^ (((B ^ (A >> n)) & M) << n), B ^ ((B ^ (A >> n)) & M))

    def rowperm(self,S, B0_pos, B1_pos, B2_pos, B3_pos):
        T = 0x0000;
        for b in range(0, 4):
            T |= ((S >> (4 * b + 0)) & 0x1) << (b + 4 * B0_pos);
            T |= ((S >> (4 * b + 1)) & 0x1) << (b + 4 * B1_pos);
            T |= ((S >> (4 * b + 2)) & 0x1) << (b + 4 * B2_pos);
            T |= ((S >> (4 * b + 3)) & 0x1) << (b + 4 * B3_pos);


        return (T)

    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        c = np.copy(p);
        # ===SubCells===#
        c[1] ^= c[0] & c[2];
        c[0] ^= c[1] & c[3];
        c[2] ^= c[0] | c[1];
        c[3] ^= c[2];
        c[1] ^= c[3];
        c[3] ^= 0xffff;
        c[2] ^= c[0] & c[1];
        T = np.copy(c[0]);
        c[0] = np.copy(c[3]);
        c[3] = np.copy(T);
        # ===PermBits===#
        c[0] = self.rowperm(c[0], 0, 3, 2, 1);
        c[1] = self.rowperm(c[1], 1, 0, 3, 2);
        c[2] = self.rowperm(c[2], 2, 1, 0, 3);
        c[3] = self.rowperm(c[3], 3, 2, 1, 0);
        # ===AddRoundKey===#
        c[1] ^= k[6];
        c[0] ^= k[7];
        # Add round constant#
        c[3] ^= 0x8000 ^ rc;
        return c

    def encrypt(self, p, ks,r_start=1):
        """
        Encrypt by applying the round function for each given round key
        :param p: The plaintext
        :param keys: A list of round keys
        :return: The encryption of p under the round keys in keys
        """
        # For Present, the number of round keys is the number of rounds + 1
        c = np.copy(p[::-1]);
        for i in range(0, 4): c[i] = self.bit_at_pos(c[i], 0x0a0a, 3);
        for i in range(0, 4): c[i] = self.bit_at_pos(c[i], 0x00cc, 6);
        for i in range(1, 4): c[0], c[i] = self.bit_to_pos(c[0], c[i], 0x000f, 4 * i);
        for i in range(2, 4): c[1], c[i] = self.bit_to_pos(c[1], c[i], 0x00f0, 4 * (i - 1));
        for i in range(3, 4): c[2], c[i] = self.bit_to_pos(c[2], c[i], 0x0f00, 4 * (i - 2));
        c0, c1, c2, c3 = c[0], c[1], c[2], c[3];
        for i in range(r_start-1, len(ks)):
            (c0, c1, c2, c3) = self.encrypt_one_round((c0, c1, c2, c3), ks[i], self.GIFT_RC[i])


        c = np.array([c0, c1, c2, c3], dtype=np.uint16);
        for i in range(3, 4): c[2], c[i] = self.bit_to_pos(c[2], c[i], 0x0f00, 4 * (i - 2));
        for i in range(2, 4): c[1], c[i] = self.bit_to_pos(c[1], c[i], 0x00f0, 4 * (i - 1));
        for i in range(1, 4): c[0], c[i] = self.bit_to_pos(c[0], c[i], 0x000f, 4 * i);
        for i in range(0, 4): c[i] = self.bit_at_pos(c[i], 0x00cc, 6);
        for i in range(0, 4): c[i] = self.bit_at_pos(c[i], 0x0a0a, 3);
        return (c[::-1]);

    def rowperm_dec(self,S, B_pos):
        T = 0x0000;
        for b in range(0, 4):
            T |= ((S >> (4 * b + 0)) & 0x1) << (4 * 0 + B_pos[b]);
            T |= ((S >> (4 * b + 1)) & 0x1) << (4 * 1 + B_pos[b]);
            T |= ((S >> (4 * b + 2)) & 0x1) << (4 * 2 + B_pos[b]);
            T |= ((S >> (4 * b + 3)) & 0x1) << (4 * 3 + B_pos[b]);


        return (T)

    def decrypt_one_round(self, c, k, rc=None):
        """
        Inverse round function of the cipher
        :param c: The ciphertext
        :param k: The round key
        :param rc: The round constant
        :return: The one round decryption of c using key k
        """
        p = np.copy(c);
        # Add round constant#
        p[3] ^= 0x8000 ^ rc;
        # ===AddRoundKey===#
        p[1] ^= k[6];
        p[0] ^= k[7];
        # ===PermBits===#
        p[0] = rowperm_dec(p[0], [0, 3, 2, 1]);
        p[1] = rowperm_dec(p[1], [1, 0, 3, 2]);
        p[2] = rowperm_dec(p[2], [2, 1, 0, 3]);
        p[3] = rowperm_dec(p[3], [3, 2, 1, 0]);
        # ===SubCells===#
        T = np.copy(p[0]);
        p[0] = np.copy(p[3]);
        p[3] = np.copy(T);
        p[2] ^= p[0] & p[1];
        p[3] ^= 0xffffffff;
        p[1] ^= p[3];
        p[3] ^= p[2];
        p[2] ^= p[0] | p[1];
        p[0] ^= p[1] & p[3];
        p[1] ^= p[0] & p[2];
        return p

    def decrypt(self, c, ks,r_start=28,r_end=0):
        """
        Decrypt by applying the inverse round function for each given key
        :param c: The ciphertext
        :param keys: A list of round keys
        :return: The decryption of c under the round keys in keys
        """
        p = np.copy(c[::-1]);
        for i in range(0, 4): p[i] =self.bit_at_pos(p[i], 0x0a0a, 3);
        for i in range(0, 4): p[i] =self.bit_at_pos(p[i], 0x00cc, 6);
        for i in range(1, 4): p[0], p[i] = self.bit_to_pos(p[0], p[i], 0x000f, 4 * i);
        for i in range(2, 4): p[1], p[i] = self.bit_to_pos(p[1], p[i], 0x00f0, 4 * (i - 1));
        for i in range(3, 4): p[2], p[i] = self.bit_to_pos(p[2], p[i], 0x0f00, 4 * (i - 2));
        c0, c1, c2, c3 = p[0], p[1], p[2], p[3];
        for i in range(r_start-1,r_end-1,-1):
            (c0, c1, c2, c3) = decrypt_one_round((c0, c1, c2, c3), ks[i], GIFT_RC[i])
        p = np.array([c0, c1, c2, c3], dtype=np.uint16);
        for i in range(3, 4): p[2], p[i] = self.bit_to_pos(p[2], p[i], 0x0f00, 4 * (i - 2));
        for i in range(2, 4): p[1], p[i] = self.bit_to_pos(p[1], p[i], 0x00f0, 4 * (i - 1));
        for i in range(1, 4): p[0], p[i] = self.bit_to_pos(p[0], p[i], 0x000f, 4 * i);
        for i in range(0, 4): p[i] = self.bit_at_pos(S[i], 0x00cc, 6);
        for i in range(0, 4): p[i] =self.bit_at_pos(S[i], 0x0a0a, 3);
        return p[::-1]

    def calc_back(self, c, p=None, variant=1):
        """
        Revert deterministic parts of the round function
        :param c: The ciphertext
        :param p: The initial plaintext
        :param variant: Select the variant of how to calculate back (default is 1; 0 means not calculating back)
        :return: The inner state after reverting the deterministic transformation at the end of the encryption process
        """
        if variant == 0:
            return c
        raise Exception("ERROR: No variant of calculating back is implemented")

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        W = np.copy(key);
        ks = [0 for i in range(self.n_rounds)];
        ks[0] = W.copy();
        for i in range(1, self.n_rounds):
            T6 = (W[6] >> 2) | (W[6] << 14);
            T7 = (W[7] >> 12) | (W[7] << 4);
            W[7] = W[5];
            W[6] = W[4];
            W[5] = W[3];
            W[4] = W[2];
            W[3] = W[1];
            W[2] = W[0];
            W[1] = T7;
            W[0] = T6;
            ks[i] = np.copy(W);


        return (ks);

    @staticmethod
    def get_test_vectors():
        gift_64 = Gift()
        key = [0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000]
        ks = gift_64.key_schedule(key)
        pt = [0x0000, 0x0000, 0x0000, 0x0000]
        ct = [0xf62b, 0xc3ef, 0x34f7, 0x75ac]
        return [(gift_64, pt, ks, ct)]





