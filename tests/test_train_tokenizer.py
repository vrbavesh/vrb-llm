import unittest

from train_tokenizer import clean_book, split_books


class PreparationTests(unittest.TestCase):
    def test_removes_only_marked_boilerplate(self):
        body = "  A book.\n\nSecond paragraph.\n"
        raw = "Header\n*** START OF THE PROJECT GUTENBERG EBOOK TITLE ***\n" + body + "*** END OF THIS PROJECT GUTENBERG EBOOK TITLE ***\nFooter"
        self.assertEqual(clean_book(raw), (body, True, True))
        self.assertEqual(clean_book(body), (body, False, False))

    def test_one_missing_marker(self):
        self.assertEqual(clean_book("Header\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nBody"), ("Body", True, False))
        self.assertEqual(clean_book("Body\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nFooter"), ("Body\n", False, True))

    def test_split_is_disjoint_complete_and_order_independent(self):
        ids = [str(i) for i in range(100)]
        train, test = split_books(ids)
        self.assertEqual((len(train), len(test)), (80, 20))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train + test), set(ids))
        self.assertEqual(split_books(list(reversed(ids))), (train, test))
        self.assertEqual(split_books(ids + [ids[0], ids[0] + "-8"]), (train, test))
        self.assertEqual(set(sum(split_books(["phys-0", "1"]), [])), {"phys", "1"})
        with self.assertRaises(ValueError):
            split_books(["", "1"])


if __name__ == "__main__":
    unittest.main()
