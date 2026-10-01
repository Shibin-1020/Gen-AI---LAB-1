# Failure candidates - gpt_char_v1_20261001-153007

Automatically ranked from `samples.json`. Read them, pick three genuine failures, and write them up in `failure_analysis.md`.

## Repetition (highest repeated-4-gram rate)

- **'In a small house near the forest,' / greedy** - rep4=0.61, repeats=[('the tree was very', 16), ('tree was very high', 13), ('and the tree was', 7)], OOV=[], names=[], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  In a small house near the forest, there was a big tree. The tree was very high and the tree was very high. The tree was very high up in the tree. The tree was very high and the tree was very high.
  
  One day, the tree saw a big tree. The tree was very high and the tree was very high. The tree was very high and the tree was very high. The tree was very high and the tree was very high. The tree was very high and the tree was very high.
  
  The tree was very happy to see the tree and the tree was very happy. The tree was very proud of 
  ```

- **'There was a big red ball' / greedy** - rep4=0.34, repeats=[('the ball was sad', 5), ('ball was sad and', 5), ('wanted to play with', 5)], OOV=[], names=[], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  There was a big red ball in the park. It was so big and shiny and had a long tail. The ball was so happy to see it.
  
  But then, a big wind came and the ball went away. The ball was sad and didn't know what to do. The ball was sad and wanted to play with it. The ball wanted to play with the ball and the ball. The ball was sad and wanted to play with the ball.
  
  The ball was sad and didn't know what to do. The ball was sad and wanted to help the ball. The ball wanted to play with the ball too. The ball wanted to play with 
  ```

- **'The sun was shining and' / greedy** - rep4=0.16, repeats=[('the bird was so', 2), ('bird was so happy', 2), ('the bird said thank', 2)], OOV=['togethe'], names=['Thank'], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  The sun was shining and the sky was blue. The bird was so happy to see the sun shine brightly in the sky. The sun was shining and the bird was so happy.
  
  The bird said, "Thank you for helping me find my way home. I will fly again soon."
  
  The bird smiled and said, "You're welcome, bird. I'm glad you are so kind."
  
  The bird said, "Thank you for helping me find my way home."
  
  The bird said, "You're welcome, bird. I'm glad you like it. Now let's go home and play with your friends."
  
  The bird and the bird went home togethe
  ```

- **'The cat saw a bird and' / greedy** - rep4=0.13, repeats=[('the bird said you', 3), ('bird said you are', 3), ('want to play with', 2)], OOV=[], names=['Hello', 'Thank'], no_eos=True, doubled=[], unbalanced_quotes=True

  ```text
  The cat saw a bird and said, "Hello, bird. Do you want to play with me?" The bird said, "Yes, I want to play with you." The bird said, "You are so cute and fluffy. I will play with you." The bird said, "Thank you, bird. You are very nice." The bird said, "You are welcome, but you are also very kind." The bird said, "You are very kind and smart. I love you too." The bird said, "I love you too, but you have to be careful. I will be careful and stay with me." The bird said, "I will be careful not to be careless and stay
  ```

## Broken / invented words (most out-of-vocabulary words)

- **'In a small house near the forest,' / temp0.8 #2** - rep4=0.00, repeats=[], OOV=['lute', 'luter', 'lutte', 'lutter'], names=['Can', 'Tim'], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  In a small house near the forest, there was a big field. The thief had fallen over the cart.
  
  "Can you help me with my tooth?" asked Tim.
  
  Tim was scared. He ran and ran until he came across a big lute. He saw a small tree which was very pretty. Tim wanted to take the lutter from him. He said yes, but he had to be careful.
  
  The luter was big and boring. Tim saw that the lutte was scary and hurt. He wanted to help his friends so he could move to the lutter which was his favorite thing. After that, Tim learned that being a lute a
  ```

- **'Once upon a time' / temp0.8 #1** - rep4=0.01, repeats=[('and the ship was', 2)], OOV=['catched', 'percil', 'percy'], names=['Percil', 'Percy'], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  Once upon a time, there was a fisherman named Percil. Percil was an only fish, but he was very powerful. He had found a special shine in the sea in the sea.
  
  Percy knew that he had not known an idea. He went to the sea and catched it with its feathers. He took it on the shore and then he saw the big ship on the way. The ship opened up and the ship was filled with the mountain. Perhaps came and the ship was beautiful like stars.
  
  The brave child was so excited, even though it was getting all the way to the ship.
  ```

- **'One day, a little girl named Lily' / temp0.8 #0** - rep4=0.00, repeats=[], OOV=['dis', 'plumbers'], names=[], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  One day, a little girl named Lily went to the park to play with her friends. The park was funny and she saw a big puddle of water. She decided to take the puddle home and splash into it.
  
  When she got to the puddle, she saw lots of plumbers. But when she did this exam, the puddle splashed around her waist. The plumber splashed around in the puddles, and the water splashing in the water. Lily was scared, but she knew she had to get back to the plumber every day. She saw the plumber and thought, so she started to lay down and dis
  ```

- **'Ben was sad because' / temp0.8 #0** - rep4=0.00, repeats=[], OOV=['polared', 'sparred'], names=['Mia'], no_eos=False, doubled=[], unbalanced_quotes=False

  ```text
  Ben was sad because he did not put on his new dress. He looked at it and said, "I'm sorry, but I made a special dress. Can you try it?"
  
  Mia nodded and ran to the grass. She saw a dragon on the ground and scared it under the west. She saw a butterfly and polared it until it sparred again. She ran back to the window and saw a bird in a tree. The bird was happy songs and now could fly again.
  ```

## Loss of coherence (most character names; no <eos>)

- **'After school, Sam wanted to' / temp0.8 #1** - rep4=0.00, repeats=[], OOV=['hom', 'tremommeter'], names=['Jack', 'Sam', 'Soon', 'Together', 'Tom'], no_eos=True, doubled=[], unbalanced_quotes=False

  ```text
  After school, Sam wanted to go to the zoo and see all the animals.
  
  They all walked closer and held on tight. Sam thought it was tremommeter than the diamond. They all had fun playing hide and seek, and Sam wanted to eat the diamond he really wanted.
  
  Together, Jack and Sam started to fill their bucket with food. He was so proud of their work!
  
  Soon, their mother came and they were all so happy. The town was perfect, and they had the best place ever.
  
  Tom and Sam continued to throw their bucket until it was time to go hom
  ```

- **'Once upon a time' / temp0.8 #2** - rep4=0.00, repeats=[], OOV=[], names=['From', 'Lily', 'Wow'], no_eos=True, doubled=['candy', 'candy'], unbalanced_quotes=False

  ```text
  Once upon a time, there was a little girl named Lily. She loved to play in her room with her dolls. One day, she found a big basket of candy and she thought it looked so yummy.
  
  Lily's mom said, "Wow, that's a good choice. Let's drink some tasty candy!" Lily smiled and nodded at Lily's hand. They walked to the basket and got all wet. Lily felt proud of herself for doing the right thing and decided to wash the candy candy.
  
  From that day on, Lily and the candy puzzled candy candy can let bad grown to be fun. The
  ```

- **'Tom and his dog went to the park.' / temp0.8 #1** - rep4=0.01, repeats=[('took the sandwich and', 2)], OOV=['ama'], names=['Ben', 'Lily', 'Wow'], no_eos=True, doubled=[], unbalanced_quotes=True

  ```text
  Tom and his dog went to the park. They saw a big tree. It was a sandwich. The sandwich made them laugh. The sandwich was not funny. It was fun and hard. Lily wanted to attach the sandwich.
  
  "Let's make the sandwich a treat!" Mom said. She took the sandwich and put it on a head.
  
  Ben and Lily were very happy. They took the sandwich and went to the store. They bought shells and toys and books. They cut the box and used the sandwich to dip it in extra spun.
  
  Mom came out and saw the sandwich. She smiled and said, "Wow, this is ama
  ```

- **'Mom said, "' / greedy** - rep4=0.02, repeats=[('to the store and', 2), ('lily saw a big', 2)], OOV=[], names=['As', 'Lily', 'Thank'], no_eos=True, doubled=[], unbalanced_quotes=True

  ```text
  Mom said, "Let's go to the store and buy some cookies."
  
  Lily was so excited that she ran to the store and bought a cookie. She was so happy and said, "Thank you, Mom!"
  
  At the store, Lily saw a big cookie in the kitchen. She asked her mom if she could have it. Her mom said yes, but she had to be careful when she was done. Lily was sad because she wanted to help her mom.
  
  As they were walking through the kitchen, Lily saw a big box of cookies. She wanted to see if they could help her mom with the cookies. 
  ```

## Broken grammar (long sentences, doubled words, unbalanced quotes)

- **'Once upon a time' / temp0.8 #2** - rep4=0.00, repeats=[], OOV=[], names=['From', 'Lily', 'Wow'], no_eos=True, doubled=['candy', 'candy'], unbalanced_quotes=False

  ```text
  Once upon a time, there was a little girl named Lily. She loved to play in her room with her dolls. One day, she found a big basket of candy and she thought it looked so yummy.
  
  Lily's mom said, "Wow, that's a good choice. Let's drink some tasty candy!" Lily smiled and nodded at Lily's hand. They walked to the basket and got all wet. Lily felt proud of herself for doing the right thing and decided to wash the candy candy.
  
  From that day on, Lily and the candy puzzled candy candy can let bad grown to be fun. The
  ```

- **'Once upon a time' / greedy** - rep4=0.06, repeats=[('to pick it up', 2), ('pick it up and', 2), ('it up and put', 2)], OOV=[], names=['Hello', 'Lily'], no_eos=True, doubled=[], unbalanced_quotes=True

  ```text
  Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big bug on the ground. She wanted to pick it up and put it in the sunshine.
  
  Lily went to the bug and said, "Hello, bug! What is that?" The bug didn't know what to do. Lily was sad because she couldn't pick it up.
  
  But then, Lily's mom came in and said, "Lily, you can't pick the bug anymore. It's too big for you." Lily felt sad and said, "But I want to pick it up and put it in the bucket to make i
  ```

- **'Tom and his dog went to the park.' / temp0.8 #1** - rep4=0.01, repeats=[('took the sandwich and', 2)], OOV=['ama'], names=['Ben', 'Lily', 'Wow'], no_eos=True, doubled=[], unbalanced_quotes=True

  ```text
  Tom and his dog went to the park. They saw a big tree. It was a sandwich. The sandwich made them laugh. The sandwich was not funny. It was fun and hard. Lily wanted to attach the sandwich.
  
  "Let's make the sandwich a treat!" Mom said. She took the sandwich and put it on a head.
  
  Ben and Lily were very happy. They took the sandwich and went to the store. They bought shells and toys and books. They cut the box and used the sandwich to dip it in extra spun.
  
  Mom came out and saw the sandwich. She smiled and said, "Wow, this is ama
  ```

- **'Mom said, "' / greedy** - rep4=0.02, repeats=[('to the store and', 2), ('lily saw a big', 2)], OOV=[], names=['As', 'Lily', 'Thank'], no_eos=True, doubled=[], unbalanced_quotes=True

  ```text
  Mom said, "Let's go to the store and buy some cookies."
  
  Lily was so excited that she ran to the store and bought a cookie. She was so happy and said, "Thank you, Mom!"
  
  At the store, Lily saw a big cookie in the kitchen. She asked her mom if she could have it. Her mom said yes, but she had to be careful when she was done. Lily was sad because she wanted to help her mom.
  
  As they were walking through the kitchen, Lily saw a big box of cookies. She wanted to see if they could help her mom with the cookies. 
  ```

