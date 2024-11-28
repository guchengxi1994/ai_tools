#[allow(dead_code, unused_variables)]
mod magika_test;
#[allow(dead_code, unused_imports, unused_variables)]
mod mlp_test;
#[macro_use]
pub mod macros;
mod command;
mod gan;
mod reflection_pad2d;
mod resnet_generator_test;
mod unet;

#[allow(unused_imports, dead_code, unused_variables)]
mod tests {
    use std::any;
    use std::mem::forget;

    use candle_core::cuda::cudarc::driver::result::device;
    use candle_core::{Device, Result, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::generation::LogitsProcessor;
    use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
    use image::{DynamicImage, ImageReader};
    use tokenizers::Tokenizer;

    pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
    pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

    pub const YOLO_CLASSES: [&str; 80] = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ];

    pub const CLASSES: [&str; 1000] = [
    "tench, Tinca tinca",
    "goldfish, Carassius auratus",
    "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "tiger shark, Galeocerdo cuvieri",
    "hammerhead, hammerhead shark",
    "electric ray, crampfish, numbfish, torpedo",
    "stingray",
    "cock",
    "hen",
    "ostrich, Struthio camelus",
    "brambling, Fringilla montifringilla",
    "goldfinch, Carduelis carduelis",
    "house finch, linnet, Carpodacus mexicanus",
    "junco, snowbird",
    "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "robin, American robin, Turdus migratorius",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel, dipper",
    "kite",
    "bald eagle, American eagle, Haliaeetus leucocephalus",
    "vulture",
    "great grey owl, great gray owl, Strix nebulosa",
    "European fire salamander, Salamandra salamandra",
    "common newt, Triturus vulgaris",
    "eft",
    "spotted salamander, Ambystoma maculatum",
    "axolotl, mud puppy, Ambystoma mexicanum",
    "bullfrog, Rana catesbeiana",
    "tree frog, tree-frog",
    "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "loggerhead, loggerhead turtle, Caretta caretta",
    "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "mud turtle",
    "terrapin",
    "box turtle, box tortoise",
    "banded gecko",
    "common iguana, iguana, Iguana iguana",
    "American chameleon, anole, Anolis carolinensis",
    "whiptail, whiptail lizard",
    "agama",
    "frilled lizard, Chlamydosaurus kingi",
    "alligator lizard",
    "Gila monster, Heloderma suspectum",
    "green lizard, Lacerta viridis",
    "African chameleon, Chamaeleo chamaeleon",
    "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "African crocodile, Nile crocodile, Crocodylus niloticus",
    "American alligator, Alligator mississipiensis",
    "triceratops",
    "thunder snake, worm snake, Carphophis amoenus",
    "ringneck snake, ring-necked snake, ring snake",
    "hognose snake, puff adder, sand viper",
    "green snake, grass snake",
    "king snake, kingsnake",
    "garter snake, grass snake",
    "water snake",
    "vine snake",
    "night snake, Hypsiglena torquata",
    "boa constrictor, Constrictor constrictor",
    "rock python, rock snake, Python sebae",
    "Indian cobra, Naja naja",
    "green mamba",
    "sea snake",
    "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "sidewinder, horned rattlesnake, Crotalus cerastes",
    "trilobite",
    "harvestman, daddy longlegs, Phalangium opilio",
    "scorpion",
    "black and gold garden spider, Argiope aurantia",
    "barn spider, Araneus cavaticus",
    "garden spider, Aranea diademata",
    "black widow, Latrodectus mactans",
    "tarantula",
    "wolf spider, hunting spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse, partridge, Bonasa umbellus",
    "prairie chicken, prairie grouse, prairie fowl",
    "peacock",
    "quail",
    "partridge",
    "African grey, African gray, Psittacus erithacus",
    "macaw",
    "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser, Mergus serrator",
    "goose",
    "black swan, Cygnus atratus",
    "tusker",
    "echidna, spiny anteater, anteater",
    "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
    "wallaby, brush kangaroo",
    "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "wombat",
    "jellyfish",
    "sea anemone, anemone",
    "brain coral",
    "flatworm, platyhelminth",
    "nematode, nematode worm, roundworm",
    "conch",
    "snail",
    "slug",
    "sea slug, nudibranch",
    "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "chambered nautilus, pearly nautilus, nautilus",
    "Dungeness crab, Cancer magister",
    "rock crab, Cancer irroratus",
    "fiddler crab",
    "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "crayfish, crawfish, crawdad, crawdaddy",
    "hermit crab",
    "isopod",
    "white stork, Ciconia ciconia",
    "black stork, Ciconia nigra",
    "spoonbill",
    "flamingo",
    "little blue heron, Egretta caerulea",
    "American egret, great white heron, Egretta albus",
    "bittern",
    "crane",
    "limpkin, Aramus pictus",
    "European gallinule, Porphyrio porphyrio",
    "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "bustard",
    "ruddy turnstone, Arenaria interpres",
    "red-backed sandpiper, dunlin, Erolia alpina",
    "redshank, Tringa totanus",
    "dowitcher",
    "oystercatcher, oyster catcher",
    "pelican",
    "king penguin, Aptenodytes patagonica",
    "albatross, mollymawk",
    "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
    "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
    "dugong, Dugong dugon",
    "sea lion",
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog, Maltese terrier, Maltese",
    "Pekinese, Pekingese, Peke",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound, Afghan",
    "basset, basset hound",
    "beagle",
    "bloodhound, sleuthhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound, Walker foxhound",
    "English foxhound",
    "redbone",
    "borzoi, Russian wolfhound",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound, Ibizan Podenco",
    "Norwegian elkhound, elkhound",
    "otterhound, otter hound",
    "Saluki, gazelle hound",
    "Scottish deerhound, deerhound",
    "Weimaraner",
    "Staffordshire bullterrier, Staffordshire bull terrier",
    "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier, Sealyham",
    "Airedale, Airedale terrier",
    "cairn, cairn terrier",
    "Australian terrier",
    "Dandie Dinmont, Dandie Dinmont terrier",
    "Boston bull, Boston terrier",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier, Scottish terrier, Scottie",
    "Tibetan terrier, chrysanthemum dog",
    "silky terrier, Sydney silky",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa, Lhasa apso",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla, Hungarian pointer",
    "English setter",
    "Irish setter, red setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber, clumber spaniel",
    "English springer, English springer spaniel",
    "Welsh springer spaniel",
    "cocker spaniel, English cocker spaniel, cocker",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog, bobtail",
    "Shetland sheepdog, Shetland sheep dog, Shetland",
    "collie",
    "Border collie",
    "Bouvier des Flandres, Bouviers des Flandres",
    "Rottweiler",
    "German shepherd, German shepherd dog, German police dog, alsatian",
    "Doberman, Doberman pinscher",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard, St Bernard",
    "Eskimo dog, husky",
    "malamute, malemute, Alaskan malamute",
    "Siberian husky",
    "dalmatian, coach dog, carriage dog",
    "affenpinscher, monkey pinscher, monkey dog",
    "basenji",
    "pug, pug-dog",
    "Leonberg",
    "Newfoundland, Newfoundland dog",
    "Great Pyrenees",
    "Samoyed, Samoyede",
    "Pomeranian",
    "chow, chow chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke, Pembroke Welsh corgi",
    "Cardigan, Cardigan Welsh corgi",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "timber wolf, grey wolf, gray wolf, Canis lupus",
    "white wolf, Arctic wolf, Canis lupus tundrarum",
    "red wolf, maned wolf, Canis rufus, Canis niger",
    "coyote, prairie wolf, brush wolf, Canis latrans",
    "dingo, warrigal, warragal, Canis dingo",
    "dhole, Cuon alpinus",
    "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "hyena, hyaena",
    "red fox, Vulpes vulpes",
    "kit fox, Vulpes macrotis",
    "Arctic fox, white fox, Alopex lagopus",
    "grey fox, gray fox, Urocyon cinereoargenteus",
    "tabby, tabby cat",
    "tiger cat",
    "Persian cat",
    "Siamese cat, Siamese",
    "Egyptian cat",
    "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "lynx, catamount",
    "leopard, Panthera pardus",
    "snow leopard, ounce, Panthera uncia",
    "jaguar, panther, Panthera onca, Felis onca",
    "lion, king of beasts, Panthera leo",
    "tiger, Panthera tigris",
    "cheetah, chetah, Acinonyx jubatus",
    "brown bear, bruin, Ursus arctos",
    "American black bear, black bear, Ursus americanus, Euarctos americanus",
    "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
    "sloth bear, Melursus ursinus, Ursus ursinus",
    "mongoose",
    "meerkat, mierkat",
    "tiger beetle",
    "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "ground beetle, carabid beetle",
    "long-horned beetle, longicorn, longicorn beetle",
    "leaf beetle, chrysomelid",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant, emmet, pismire",
    "grasshopper, hopper",
    "cricket",
    "walking stick, walkingstick, stick insect",
    "cockroach, roach",
    "mantis, mantid",
    "cicada, cicala",
    "leafhopper",
    "lacewing, lacewing fly",
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "damselfly",
    "admiral",
    "ringlet, ringlet butterfly",
    "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "cabbage butterfly",
    "sulphur butterfly, sulfur butterfly",
    "lycaenid, lycaenid butterfly",
    "starfish, sea star",
    "sea urchin",
    "sea cucumber, holothurian",
    "wood rabbit, cottontail, cottontail rabbit",
    "hare",
    "Angora, Angora rabbit",
    "hamster",
    "porcupine, hedgehog",
    "fox squirrel, eastern fox squirrel, Sciurus niger",
    "marmot",
    "beaver",
    "guinea pig, Cavia cobaya",
    "sorrel",
    "zebra",
    "hog, pig, grunter, squealer, Sus scrofa",
    "wild boar, boar, Sus scrofa",
    "warthog",
    "hippopotamus, hippo, river horse, Hippopotamus amphibius",
    "ox",
    "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
    "bison",
    "ram, tup",
    "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "ibex, Capra ibex",
    "hartebeest",
    "impala, Aepyceros melampus",
    "gazelle",
    "Arabian camel, dromedary, Camelus dromedarius",
    "llama",
    "weasel",
    "mink",
    "polecat, fitch, foulmart, foumart, Mustela putorius",
    "black-footed ferret, ferret, Mustela nigripes",
    "otter",
    "skunk, polecat, wood pussy",
    "badger",
    "armadillo",
    "three-toed sloth, ai, Bradypus tridactylus",
    "orangutan, orang, orangutang, Pongo pygmaeus",
    "gorilla, Gorilla gorilla",
    "chimpanzee, chimp, Pan troglodytes",
    "gibbon, Hylobates lar",
    "siamang, Hylobates syndactylus, Symphalangus syndactylus",
    "guenon, guenon monkey",
    "patas, hussar monkey, Erythrocebus patas",
    "baboon",
    "macaque",
    "langur",
    "colobus, colobus monkey",
    "proboscis monkey, Nasalis larvatus",
    "marmoset",
    "capuchin, ringtail, Cebus capucinus",
    "howler monkey, howler",
    "titi, titi monkey",
    "spider monkey, Ateles geoffroyi",
    "squirrel monkey, Saimiri sciureus",
    "Madagascar cat, ring-tailed lemur, Lemur catta",
    "indri, indris, Indri indri, Indri brevicaudatus",
    "Indian elephant, Elephas maximus",
    "African elephant, Loxodonta africana",
    "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
    "barracouta, snoek",
    "eel",
    "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
    "rock beauty, Holocanthus tricolor",
    "anemone fish",
    "sturgeon",
    "gar, garfish, garpike, billfish, Lepisosteus osseus",
    "lionfish",
    "puffer, pufferfish, blowfish, globefish",
    "abacus",
    "abaya",
    "academic gown, academic robe, judge's robe",
    "accordion, piano accordion, squeeze box",
    "acoustic guitar",
    "aircraft carrier, carrier, flattop, attack aircraft carrier",
    "airliner",
    "airship, dirigible",
    "altar",
    "ambulance",
    "amphibian, amphibious vehicle",
    "analog clock",
    "apiary, bee house",
    "apron",
    "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
    "assault rifle, assault gun",
    "backpack, back pack, knapsack, packsack, rucksack, haversack",
    "bakery, bakeshop, bakehouse",
    "balance beam, beam",
    "balloon",
    "ballpoint, ballpoint pen, ballpen, Biro",
    "Band Aid",
    "banjo",
    "bannister, banister, balustrade, balusters, handrail",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel, cask",
    "barrow, garden cart, lawn cart, wheelbarrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "bathing cap, swimming cap",
    "bath towel",
    "bathtub, bathing tub, bath, tub",
    "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    "beacon, lighthouse, beacon light, pharos",
    "beaker",
    "bearskin, busby, shako",
    "beer bottle",
    "beer glass",
    "bell cote, bell cot",
    "bib",
    "bicycle-built-for-two, tandem bicycle, tandem",
    "bikini, two-piece",
    "binder, ring-binder",
    "binoculars, field glasses, opera glasses",
    "birdhouse",
    "boathouse",
    "bobsled, bobsleigh, bob",
    "bolo tie, bolo, bola tie, bola",
    "bonnet, poke bonnet",
    "bookcase",
    "bookshop, bookstore, bookstall",
    "bottlecap",
    "bow",
    "bow tie, bow-tie, bowtie",
    "brass, memorial tablet, plaque",
    "brassiere, bra, bandeau",
    "breakwater, groin, groyne, mole, bulwark, seawall, jetty",
    "breastplate, aegis, egis",
    "broom",
    "bucket, pail",
    "buckle",
    "bulletproof vest",
    "bullet train, bullet",
    "butcher shop, meat market",
    "cab, hack, taxi, taxicab",
    "caldron, cauldron",
    "candle, taper, wax light",
    "cannon",
    "canoe",
    "can opener, tin opener",
    "cardigan",
    "car mirror",
    "carousel, carrousel, merry-go-round, roundabout, whirligig",
    "carpenter's kit, tool kit",
    "carton",
    "car wheel",
    "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello, violoncello",
    "cellular telephone, cellular phone, cellphone, cell, mobile phone",
    "chain",
    "chainlink fence",
    "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
    "chain saw, chainsaw",
    "chest",
    "chiffonier, commode",
    "chime, bell, gong",
    "china cabinet, china closet",
    "Christmas stocking",
    "church, church building",
    "cinema, movie theater, movie theatre, movie house, picture palace",
    "cleaver, meat cleaver, chopper",
    "cliff dwelling",
    "cloak",
    "clog, geta, patten, sabot",
    "cocktail shaker",
    "coffee mug",
    "coffeepot",
    "coil, spiral, volute, whorl, helix",
    "combination lock",
    "computer keyboard, keypad",
    "confectionery, confectionary, candy store",
    "container ship, containership, container vessel",
    "convertible",
    "corkscrew, bottle screw",
    "cornet, horn, trumpet, trump",
    "cowboy boot",
    "cowboy hat, ten-gallon hat",
    "cradle",
    "crane",
    "crash helmet",
    "crate",
    "crib, cot",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam, dike, dyke",
    "desk",
    "desktop computer",
    "dial telephone, dial phone",
    "diaper, nappy, napkin",
    "digital clock",
    "digital watch",
    "dining table, board",
    "dishrag, dishcloth",
    "dishwasher, dish washer, dishwashing machine",
    "disk brake, disc brake",
    "dock, dockage, docking facility",
    "dogsled, dog sled, dog sleigh",
    "dome",
    "doormat, welcome mat",
    "drilling platform, offshore rig",
    "drum, membranophone, tympan",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan, blower",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso maker",
    "face powder",
    "feather boa, boa",
    "file, file cabinet, filing cabinet",
    "fireboat",
    "fire engine, fire truck",
    "fire screen, fireguard",
    "flagpole, flagstaff",
    "flute, transverse flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster",
    "freight car",
    "French horn, horn",
    "frying pan, frypan, skillet",
    "fur coat",
    "garbage truck, dustcart",
    "gasmask, respirator, gas helmet",
    "gas pump, gasoline pump, petrol pump, island dispenser",
    "goblet",
    "go-kart",
    "golf ball",
    "golfcart, golf cart",
    "gondola",
    "gong, tam-tam",
    "gown",
    "grand piano, grand",
    "greenhouse, nursery, glasshouse",
    "grille, radiator grille",
    "grocery store, grocery, food market, market",
    "guillotine",
    "hair slide",
    "hair spray",
    "half track",
    "hammer",
    "hamper",
    "hand blower, blow dryer, blow drier, hair dryer, hair drier",
    "hand-held computer, hand-held microcomputer",
    "handkerchief, hankie, hanky, hankey",
    "hard disc, hard disk, fixed disk",
    "harmonica, mouth organ, harp, mouth harp",
    "harp",
    "harvester, reaper",
    "hatchet",
    "holster",
    "home theater, home theatre",
    "honeycomb",
    "hook, claw",
    "hoopskirt, crinoline",
    "horizontal bar, high bar",
    "horse cart, horse-cart",
    "hourglass",
    "iPod",
    "iron, smoothing iron",
    "jack-o'-lantern",
    "jean, blue jean, denim",
    "jeep, landrover",
    "jersey, T-shirt, tee shirt",
    "jigsaw puzzle",
    "jinrikisha, ricksha, rickshaw",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat, laboratory coat",
    "ladle",
    "lampshade, lamp shade",
    "laptop, laptop computer",
    "lawn mower, mower",
    "lens cap, lens cover",
    "letter opener, paper knife, paperknife",
    "library",
    "lifeboat",
    "lighter, light, igniter, ignitor",
    "limousine, limo",
    "liner, ocean liner",
    "lipstick, lip rouge",
    "Loafer",
    "lotion",
    "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
    "loupe, jeweler's loupe",
    "lumbermill, sawmill",
    "magnetic compass",
    "mailbag, postbag",
    "mailbox, letter box",
    "maillot",
    "maillot, tank suit",
    "manhole cover",
    "maraca",
    "marimba, xylophone",
    "mask",
    "matchstick",
    "maypole",
    "maze, labyrinth",
    "measuring cup",
    "medicine chest, medicine cabinet",
    "megalith, megalithic structure",
    "microphone, mike",
    "microwave, microwave oven",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt, mini",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home, manufactured home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "mortarboard",
    "mosque",
    "mosquito net",
    "motor scooter, scooter",
    "mountain bike, all-terrain bike, off-roader",
    "mountain tent",
    "mouse, computer mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook, notebook computer",
    "obelisk",
    "oboe, hautboy, hautbois",
    "ocarina, sweet potato",
    "odometer, hodometer, mileometer, milometer",
    "oil filter",
    "organ, pipe organ",
    "oscilloscope, scope, cathode-ray oscilloscope, CRO",
    "overskirt",
    "oxcart",
    "oxygen mask",
    "packet",
    "paddle, boat paddle",
    "paddlewheel, paddle wheel",
    "padlock",
    "paintbrush",
    "pajama, pyjama, pj's, jammies",
    "palace",
    "panpipe, pandean pipe, syrinx",
    "paper towel",
    "parachute, chute",
    "parallel bars, bars",
    "park bench",
    "parking meter",
    "passenger car, coach, carriage",
    "patio, terrace",
    "pay-phone, pay-station",
    "pedestal, plinth, footstall",
    "pencil box, pencil case",
    "pencil sharpener",
    "perfume, essence",
    "Petri dish",
    "photocopier",
    "pick, plectrum, plectron",
    "pickelhaube",
    "picket fence, paling",
    "pickup, pickup truck",
    "pier",
    "piggy bank, penny bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate, pirate ship",
    "pitcher, ewer",
    "plane, carpenter's plane, woodworking plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow, plough",
    "plunger, plumber's helper",
    "Polaroid camera, Polaroid Land camera",
    "pole",
    "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    "poncho",
    "pool table, billiard table, snooker table",
    "pop bottle, soda bottle",
    "pot, flowerpot",
    "potter's wheel",
    "power drill",
    "prayer rug, prayer mat",
    "printer",
    "prison, prison house",
    "projectile, missile",
    "projector",
    "puck, hockey puck",
    "punching bag, punch bag, punching ball, punchball",
    "purse",
    "quill, quill pen",
    "quilt, comforter, comfort, puff",
    "racer, race car, racing car",
    "racket, racquet",
    "radiator",
    "radio, wireless",
    "radio telescope, radio reflector",
    "rain barrel",
    "recreational vehicle, RV, R.V.",
    "reel",
    "reflex camera",
    "refrigerator, icebox",
    "remote control, remote",
    "restaurant, eating house, eating place, eatery",
    "revolver, six-gun, six-shooter",
    "rifle",
    "rocking chair, rocker",
    "rotisserie",
    "rubber eraser, rubber, pencil eraser",
    "rugby ball",
    "rule, ruler",
    "running shoe",
    "safe",
    "safety pin",
    "saltshaker, salt shaker",
    "sandal",
    "sarong",
    "sax, saxophone",
    "scabbard",
    "scale, weighing machine",
    "school bus",
    "schooner",
    "scoreboard",
    "screen, CRT screen",
    "screw",
    "screwdriver",
    "seat belt, seatbelt",
    "sewing machine",
    "shield, buckler",
    "shoe shop, shoe-shop, shoe store",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule, slipstick",
    "sliding door",
    "slot, one-armed bandit",
    "snorkel",
    "snowmobile",
    "snowplow, snowplough",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar dish, solar collector, solar furnace",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "speedboat",
    "spider web, spider's web",
    "spindle",
    "sports car, sport car",
    "spotlight, spot",
    "stage",
    "steam locomotive",
    "steel arch bridge",
    "steel drum",
    "stethoscope",
    "stole",
    "stone wall",
    "stopwatch, stop watch",
    "stove",
    "strainer",
    "streetcar, tram, tramcar, trolley, trolley car",
    "stretcher",
    "studio couch, day bed",
    "stupa, tope",
    "submarine, pigboat, sub, U-boat",
    "suit, suit of clothes",
    "sundial",
    "sunglass",
    "sunglasses, dark glasses, shades",
    "sunscreen, sunblock, sun blocker",
    "suspension bridge",
    "swab, swob, mop",
    "sweatshirt",
    "swimming trunks, bathing trunks",
    "swing",
    "switch, electric switch, electrical switch",
    "syringe",
    "table lamp",
    "tank, army tank, armored combat vehicle, armoured combat vehicle",
    "tape player",
    "teapot",
    "teddy, teddy bear",
    "television, television system",
    "tennis ball",
    "thatch, thatched roof",
    "theater curtain, theatre curtain",
    "thimble",
    "thresher, thrasher, threshing machine",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop, tobacconist shop, tobacconist",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck, tow car, wrecker",
    "toyshop",
    "tractor",
    "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
    "tray",
    "trench coat",
    "tricycle, trike, velocipede",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus, trolley coach, trackless trolley",
    "trombone",
    "tub, vat",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle, monocycle",
    "upright, upright piano",
    "vacuum, vacuum cleaner",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin, fiddle",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet, billfold, notecase, pocketbook",
    "wardrobe, closet, press",
    "warplane, military plane",
    "washbasin, handbasin, washbowl, lavabo, wash-hand basin",
    "washer, automatic washer, washing machine",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool, woolen, woollen",
    "worm fence, snake fence, snake-rail fence, Virginia fence",
    "wreck",
    "yawl",
    "yurt",
    "web site, website, internet site, site",
    "comic book",
    "crossword puzzle, crossword",
    "street sign",
    "traffic light, traffic signal, stoplight",
    "book jacket, dust cover, dust jacket, dust wrapper",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot, hotpot",
    "trifle",
    "ice cream, icecream",
    "ice lolly, lolly, lollipop, popsicle",
    "French loaf",
    "bagel, beigel",
    "pretzel",
    "cheeseburger",
    "hotdog, hot dog, red hot",
    "mashed potato",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini, courgette",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber, cuke",
    "artichoke, globe artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple, ananas",
    "banana",
    "jackfruit, jak, jack",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate sauce, chocolate syrup",
    "dough",
    "meat loaf, meatloaf",
    "pizza, pizza pie",
    "potpie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff, drop, drop-off",
    "coral reef",
    "geyser",
    "lakeside, lakeshore",
    "promontory, headland, head, foreland",
    "sandbar, sand bar",
    "seashore, coast, seacoast, sea-coast",
    "valley, vale",
    "volcano",
    "ballplayer, baseball player",
    "groom, bridegroom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
    "corn",
    "acorn",
    "hip, rose hip, rosehip",
    "buckeye, horse chestnut, conker",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn, carrion fungus",
    "earthstar",
    "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
    "bolete",
    "ear, spike, capitulum",
    "toilet tissue, toilet paper, bathroom tissue",
];

    struct CModel {
        first: Tensor,
        second: Tensor,
    }
    impl CModel {
        fn forward(&self, image: &Tensor) -> Result<Tensor> {
            let x = image.matmul(&self.first)?;
            let x = x.relu()?;
            x.matmul(&self.second)
        }
    }

    #[test]
    fn test() -> Result<()> {
        // Use Device::new_cuda(0)?; to use the GPU.
        let device = Device::Cpu;

        let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
        let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
        let model = CModel { first, second };

        let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

        let digit = model.forward(&dummy_image)?;
        println!("Digit {digit:?} digit");
        Ok(())
    }

    fn load_and_preprocess_image(
        path: &str,
        device: Device,
        size: usize,
        to_type: candle_core::DType,
    ) -> anyhow::Result<Tensor> {
        let img = ImageReader::open(path)?.decode()?.to_rgb8();
        let img = DynamicImage::ImageRgb8(img).resize_exact(
            size as u32,
            size as u32,
            image::imageops::FilterType::Triangle,
        );
        // 转换为张量并归一化
        let data = img.into_rgb8().into_raw();
        let data = Tensor::from_vec(data, (size, size, 3), &device)?.permute((2, 0, 1))?;

        // let r = (data.to_dtype(to_type)? / 255.)?
        //     .broadcast_sub(&mean)?
        //     .broadcast_div(&std)?;

        let r = (data.to_dtype(to_type)? / 255.)?;

        anyhow::Ok(r)
    }

    #[test]
    fn onnx_test() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let img =
            load_and_preprocess_image("assets/sock.jpg", device, 224, candle_core::DType::F32)?;

        let model = candle_onnx::read_file("assets/efficientnet-lite4-s.onnx")?;

        let graph = model.graph.as_ref().unwrap();
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(graph.input[0].name.to_string(), img.unsqueeze(0)?);

        let mut outputs = candle_onnx::simple_eval(&model, inputs)?;
        let output = outputs.remove(&graph.output[0].name).unwrap();

        let prs = candle_core::IndexOp::i(&output, 0)?.to_vec1::<f32>()?;

        let mut top: Vec<_> = prs.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = top.into_iter().take(5).collect::<Vec<_>>();

        // Print the top predictions
        for &(i, p) in &top {
            println!("{:50}: {:.2}%", CLASSES[i], p * 100.0);
        }

        anyhow::Ok(())
    }

    #[test]
    // this code is not working because candle onnx dont support maxpool
    fn onnx_yolo_test() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let img =
            load_and_preprocess_image("assets/sock.jpg", device, 640, candle_core::DType::F32)?;

        let model = candle_onnx::read_file("assets/yolov8n.onnx")?;

        let graph = model.graph.as_ref().unwrap();
        let mut inputs = std::collections::HashMap::new();
        let img = img.unsqueeze(0)?;
        println!("img shape: {:?}", img);
        inputs.insert(graph.input[0].name.to_string(), img);

        let mut outputs = candle_onnx::simple_eval(&model, inputs)?;

        let output = outputs.remove(&graph.output[0].name).unwrap();

        for i in 0..output.dims()[1] {
            let x = output.get(i)?;
            println!("{:?}", x);
        }

        anyhow::Ok(())
    }

    pub struct Model {
        pub inner: ModelForCausalLM,
    }

    impl Model {
        fn forward(&mut self, xs: &Tensor, s: usize) -> anyhow::Result<Tensor> {
            anyhow::Ok(self.inner.forward(xs, s)?)
        }
    }

    pub struct TokenOutputStream {
        tokenizer: tokenizers::Tokenizer,
        tokens: Vec<u32>,
        prev_index: usize,
        current_index: usize,
    }

    impl TokenOutputStream {
        pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
            Self {
                tokenizer,
                tokens: Vec::new(),
                prev_index: 0,
                current_index: 0,
            }
        }

        pub fn into_inner(self) -> tokenizers::Tokenizer {
            self.tokenizer
        }

        fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
            match self.tokenizer.decode(tokens, true) {
                Ok(str) => Ok(str),
                Err(err) => anyhow::bail!("cannot decode: {err}"),
            }
        }

        // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
        pub fn next_token(&mut self, token: u32) -> anyhow::Result<Option<String>> {
            let prev_text = if self.tokens.is_empty() {
                String::new()
            } else {
                let tokens = &self.tokens[self.prev_index..self.current_index];
                self.decode(tokens)?
            };
            self.tokens.push(token);
            let text = self.decode(&self.tokens[self.prev_index..])?;
            if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
                let text = text.split_at(prev_text.len());
                self.prev_index = self.current_index;
                self.current_index = self.tokens.len();
                Ok(Some(text.1.to_string()))
            } else {
                Ok(None)
            }
        }

        pub fn decode_rest(&self) -> anyhow::Result<Option<String>> {
            let prev_text = if self.tokens.is_empty() {
                String::new()
            } else {
                let tokens = &self.tokens[self.prev_index..self.current_index];
                self.decode(tokens)?
            };
            let text = self.decode(&self.tokens[self.prev_index..])?;
            if text.len() > prev_text.len() {
                let text = text.split_at(prev_text.len());
                Ok(Some(text.1.to_string()))
            } else {
                Ok(None)
            }
        }

        pub fn decode_all(&self) -> anyhow::Result<String> {
            self.decode(&self.tokens)
        }

        pub fn get_token(&self, token_s: &str) -> Option<u32> {
            self.tokenizer.get_vocab(true).get(token_s).copied()
        }

        pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
            &self.tokenizer
        }

        pub fn clear(&mut self) {
            self.tokens.clear();
            self.prev_index = 0;
            self.current_index = 0;
        }
    }

    struct TextGeneration {
        model: Model,
        device: Device,
        tokenizer: TokenOutputStream,
        logits_processor: LogitsProcessor,
        repeat_penalty: f32,
        repeat_last_n: usize,
    }

    impl TextGeneration {
        #[allow(clippy::too_many_arguments)]
        fn new(
            model: Model,
            tokenizer: Tokenizer,
            seed: u64,
            temp: Option<f64>,
            top_p: Option<f64>,
            repeat_penalty: f32,
            repeat_last_n: usize,
            device: &Device,
        ) -> Self {
            let logits_processor = LogitsProcessor::new(seed, temp, top_p);
            Self {
                model,
                tokenizer: TokenOutputStream::new(tokenizer),
                logits_processor,
                repeat_penalty,
                repeat_last_n,
                device: device.clone(),
            }
        }

        fn run(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
            use std::io::Write;
            self.tokenizer.clear();
            let mut tokens = self
                .tokenizer
                .tokenizer()
                .encode(prompt, true)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    print!("{t}")
                }
            }
            std::io::stdout().flush()?;

            let mut generated_tokens = 0usize;
            let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
                Some(token) => token,
                None => anyhow::bail!("cannot find the <|endoftext|> token"),
            };
            let start_gen = std::time::Instant::now();
            for index in 0..sample_len {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let start_pos = tokens.len().saturating_sub(context_size);
                let ctxt = &tokens[start_pos..];
                let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, start_pos)?;
                let logits = logits
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_dtype(candle_core::DType::F32)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &tokens[start_at..],
                    )?
                };

                let next_token = self.logits_processor.sample(&logits)?;
                tokens.push(next_token);
                generated_tokens += 1;
                if next_token == eos_token {
                    break;
                }
                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
            }
            let dt = start_gen.elapsed();
            if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
                print!("{rest}");
            }
            std::io::stdout().flush()?;
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
            Ok(())
        }
    }

    #[test]
    fn file_type_test() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut img = candle_core::Tensor::randn(0.5, 1.0, (1, 512), &device)?;
        img = img.to_dtype(candle_core::DType::F32)?;
        let model = candle_onnx::read_file(r"assets\converted_model.onnx")?;
        println!("model loaded");
        let graph = model.graph.as_ref().unwrap();
        let mut inputs = std::collections::HashMap::new();
        for i in 0..graph.input.len() {
            println!("input: {}", graph.input[i].name.to_string());
            println!("input : {:?}", graph.input[i].r#type);
            println!("===============================================");
        }
        inputs.insert(graph.input[0].name.to_string(), img);

        let mut outputs = candle_onnx::simple_eval(&model, inputs)?;
        let output = outputs.remove(&graph.output[0].name).unwrap();

        let prs = candle_core::IndexOp::i(&output, 0)?.to_vec1::<f32>()?;

        let mut top: Vec<_> = prs.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = top.into_iter().take(5).collect::<Vec<_>>();

        for &(i, p) in &top {
            println!("{:50}: {:.2}%", i, p * 100.0);
        }

        anyhow::Ok(())
    }

    #[test]
    fn qwen_test() -> anyhow::Result<()> {
        println!("start");
        let device = Device::cuda_if_available(0)?;
        // let model_path = "assets/Qwen2-0___5B-Instruct";
        let model_path = "assets/Qwen2___5-0___5B-Instruct";
        let model = format!("{}/model.safetensors", model_path);
        let token_file_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(token_file_path).map_err(anyhow::Error::msg)?;
        let start = std::time::Instant::now();
        let config_file_path = format!("{}/config.json", model_path);
        let config: Config = serde_json::from_slice(&std::fs::read(config_file_path)?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vec![model], candle_core::DType::F32, &device)?
        };

        println!("config loaded");

        let model = ModelForCausalLM::new(&config, vb)?;

        println!("loaded the model in {:?}", start.elapsed());

        let m = Model { inner: model };

        let mut pipeline =
            TextGeneration::new(m, tokenizer, 128, Some(0.7), Some(0.9), 1.25, 64, &device);

        let prompt = format!(
            "
        <|im_start|>system
        请用简洁、专业的语言回答问题。
        <|im_end|>

        <|im_start|>user
        {}
        <|im_end|>

        <|im_start|>assistant
        ",
            "请解释机器学习的基本概念。"
        );

        pipeline.run(&prompt, 1024)?;
        println!("end in {:?}", start.elapsed());

        anyhow::Ok(())
    }

    #[test]
    fn rwkv_test() -> anyhow::Result<()> {
        // let device = Device::Cpu;

        let device = candle_core::Device::cuda_if_available(0)?;
        println!("[rust-llm] run on device: {:?}", device);

        let mut model: crate::llm::model::Model<
            candle_transformers::models::quantized_rwkv_v6::Model,
        > = crate::llm::model::Model::<candle_transformers::models::quantized_rwkv_v6::Model>::new(
            "assets/rwkv_v6".to_string(),
        );
        crate::llm::model::ModelRun::load(&mut model)?;

        let tokenizer = candle_transformers::models::rwkv_v5::Tokenizer::new(
            model.tokenizer_path.clone().unwrap(),
        )?;

        let mut pipeline = crate::llm::text_generation::TextGeneration::<
            crate::llm::model::Model<candle_transformers::models::quantized_rwkv_v6::Model>,
            &mut candle_transformers::models::rwkv_v5::State,
            candle_transformers::models::rwkv_v5::Tokenizer,
        >::new(
            model,
            tokenizer,
            128,
            Some(0.7),
            Some(0.9),
            1.25,
            64,
            &device,
        );

        let start = std::time::Instant::now();

        let prompt = format!(
            "
        User: 你好

        Assistant: 你好。我是你的助手，我将提供专家的完整详细答复。

        User: {}

        Assistant:
        ",
            "请解释机器学习的基本概念。答案请用中文。"
        );

        pipeline.run(&prompt, 1024)?;

        println!("end in {:?}", start.elapsed());
        anyhow::Ok(())
    }
}
