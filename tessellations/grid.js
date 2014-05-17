var grid = [];
var gridRep = [];
var cellw = 100;
var cellh = 100;
var maxval = 4;
var screen = null;
var activeCells = 0;

var selected = [];

colorizer = (function() { // declare everything discreetly
  var colorizer = {};

  colorizer.scale = chroma.scale( [ '#4BF920', '#1DE5A2', '#48CC20', '#18BC49', '#0DAD6D' ] );

  colorizer.fromValue = function( v ) {
    if ( v < 0 ) return chroma('white').hex();
    else         return this.scale( v / 9 ).hex();
  }

  colorizer.highlightFromValue = function( v ) {
    if ( v < 0 ) return chroma('white').hex();
    else         return this.scale( v / 9 ).brighter().hex();
  }

  return colorizer;
})();


// translation of practical flood fill implementation as described on
// http://en.wikipedia.org/wiki/Flood_fill
floodAcquire = function( start, id ) {
  cluster = []; // this is what we return, a cluster of vector2s pointing to cells with the same id
  marked = []; // hacky way to keep track of what we've already seen
  marked.get = function( x, y ) { return this[ x + y * gridw ] === undefined ? false : this[ x + y * gridw ]; }
  marked.set = function( x, y ) { this[ x + y * gridw ] = true; }
  Q = [];
  if ( grid.get( start.x, start.y ) != id ) return [];
  Q.push( start );
  while( Q.length > 0 ) {
    var n = Q.shift();
    if ( grid.get( n.x, n.y ) == id && !marked.get( n.x, n.y ) ) {
      var w = { x: n.x, y: n.y };
      var e = { x: n.x, y: n.y };

      // move w to west until the node to the west of w is no longer id
      while( grid.get( w.x - 1, w.y ) == id ) {
        w.x--;
      }
      // move e to east until the node to the west of w is no longer id
      while( grid.get( e.x + 1, e.y ) == id ) {
        e.x++;
      }

      for ( var x = w.x; x < e.x + 1; x++ ) {
        var nn = { x: x, y: n.y };
        marked.set( nn.x, nn.y );
        cluster.push( nn );
        var north = { x: nn.x, y: nn.y - 1 };
        var south = { x: nn.x, y: nn.y + 1 };
        if ( grid.get( north.x, north.y ) == id ) Q.push( north );
        if ( grid.get( south.x, south.y ) == id ) Q.push( south );
      }
    }
  }
  return cluster;
}

draw = function() {
  screen.clear();

  // draw cells
  grid.forEach( function( e, i ) {
    var cell = screen.group()
                     .transform( { x: ( i % gridw ) * cellw, y : ( Math.floor( i / gridw ) ) * cellh } );

    cell.rect( cellw, cellh )
        .attr( { fill : colorizer.fromValue( grid[i] ) } )
        .mouseover(
          function() {
            this.fill( { color: colorizer.highlightFromValue( grid[i] ) } );
         } )
        .mouseout(
          function() {
            this.fill( { color: colorizer.fromValue( grid[i] ) } );
          }
        )
        .mousedown(
          function() {
            target = { x: i % gridw, y: Math.floor( i / gridw ) };
            selected = floodAcquire( target, grid.get( target.x, target.y ) );
            console.log( selected );
          }
        );

    var text = cell.plain( e > 0 ? e.toString() : "" )
                   .fill( { color: '#ffffff' } )
                   .transform( { x: cellw / 2, y: cellh / 2 } );

    gridRep.push( { cell : cell, text : text } );

  } );
}

update = function() {
  gridRep.forEach( function( rep, i ) {
    rep.text.plain( grid[i] > 0 ? grid[i].toString() : "" );
    rep.cell.attr( { fill : colorizer.fromValue( grid[i] ) } )
  } );
}

advance = function() {
  cellw = Math.round( cellw * 0.8 );
  cellh = Math.round( cellh * 0.8 );
  maxval++;
  init();
}

prune = function( start ) {
  // see if we should delete this cell and surrounding cells
  var targets = floodAcquire( { x: start.x, y: start.y }, grid.get( start.x, start.y ) )
  if ( targets.length == grid.get( start.x, start.y ) ) {
    targets.forEach( function( cell ) {
      grid[ cell.x + cell.y * gridw ] = -1;
      activeCells--;
      console.log( activeCells );
    } );
  }
  if ( activeCells == 0 ) {
    alert( "great job, you won!" );
    advance();
  }
}

pollDrag = function( e ) {
  var up    = false,
      down  = false,
      left  = false,
      right = false;

  if ( Math.abs( e.gesture.deltaY ) > Math.abs( e.gesture.deltaX ) ) {
    up   = e.gesture.deltaY < 0;
    down = !up;
  } else {
    left  = e.gesture.deltaX < 0;
    right = !left;
  }

  function displace( set, direction ) {
    return set.map( function( cell ) {
      return { x: cell.x + direction.x, y: cell.y + direction.y };
    } );
  }

  function checkCollision( newset, oldset ) {
    return newset.map( function( cell, i ) {
      // if cell is out of bounds, then, collision
      // if cell is not in original set and cell is not -1 then collision
      // if cell is not in original set and cell is -1 then no collision
      // if cell is in original set then no collsion
      var cellIsOutofBounds = grid.get( cell.x, cell.y ) == -2;
      var cellInOldSet = oldset.some( function( c ) { return c.x == cell.x && c.y == cell.y } );
      var isCollision = cellIsOutofBounds || ( !cellInOldSet && grid.get( cell.x, cell.y ) != -1 );
      return isCollision;
    } );
  }

  function move( from, to ) {
    // cache all the from values before clearing them
    var fromVals = from.map( function( cell ) { return grid.get( cell.x, cell.y ); } );
    from.forEach( function( cell ) { grid[ cell.x + cell.y * gridw ] = -1; } );
    to.forEach( function( cell, i ) { grid[ cell.x + cell.y * gridw ] = fromVals[i]; } );
  }

  if ( up ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy ); // oldset = newset (deep copy)
      newset = displace( oldset, { x: 0, y: -1 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  } else if ( down ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy ); // oldset = newset (deep copy)
      newset = displace( oldset, { x: 0, y: 1 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  }

  if ( left ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy );
      newset = displace( oldset, { x: -1, y: 0 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  } else if ( right ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy );
      newset = displace( oldset, { x: 1, y: 0 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  }

  if ( up || down || left || right ) {
    prune( selected[0] );
    update();
    draw();
  }
}

pollKey = function( e ) {
  e = e || window.event;
  if ( e.keyCode == 37 )        { // left
    var w = { x: selected.x, y: selected.y };

    // move w to west until the node to the west of w is no longer id
    while( grid.get( w.x - 1, w.y ) == -1 ) {
      w.x--;
    }

    // swap with w
    var temp = grid.get( w.x, w.y );
    grid[ w.x + gridw * w.y ] = grid.get( selected.x, selected.y );
    grid[ selected.x + gridw * selected.y ] = temp;

    selected = w;

    update();
    draw();
  } else if ( e.keyCode == 38 ) { // up
    var n = { x: selected.x, y: selected.y };

    // move n to norh until the node to the west of w is no longer id
    while( grid.get( n.x, n.y - 1 ) == -1 ) {
      n.y--;
    }

    // swap with n
    var temp = grid.get( n.x, n.y );
    grid[ n.x + gridw * n.y ] = grid.get( selected.x, selected.y );
    grid[ selected.x + gridw * selected.y ] = temp;

    selected = n;

    update();
    draw();
  } else if ( e.keyCode == 39 ) { // right
    var e = { x: selected.x, y: selected.y };

    // move e to east until the node to the west of w is no longer id
    while( grid.get( e.x + 1, e.y ) == -1 ) {
      e.x++;
    }

    // swap with e
    var temp = grid.get( e.x, e.y );
    grid[ e.x + gridw * e.y ] = grid.get( selected.x, selected.y );
    grid[ selected.x + gridw * selected.y ] = temp;

    selected = e;

    update();
    draw();
  } else if ( e.keyCode == 40 ) { // down
    var s = { x: selected.x, y: selected.y };

    // move s to south until the node to the west of w is no longer id
    while( grid.get( s.x, s.y + 1 ) == -1 ) {
      s.y++;
    }

    // swap with s
    var temp = grid.get( s.x, s.y );
    grid[ s.x + gridw * s.y ] = grid.get( selected.x, selected.y );
    grid[ selected.x + gridw * selected.y ] = temp;

    selected = s;

    update();
    draw();
  }
}

init = function() {
  if ( screen == null )
    screen = SVG( 'screen' ).size( 500, 500 );

  gridw = Math.floor( screen.width() / cellw ) - 1;
  gridh = Math.floor( screen.height() / cellh ) - 1;

  console.log( "new grid size is " + gridw + " " + gridh );

  var tiles = [];

  grid = [];

  // set up grid
  for ( i = 0; i < gridw * gridh; i++ ) {
    var val = Math.round( Math.random() * ( maxval - 2 ) + 2 );
    grid.push( val );
    if ( tiles[ val ] == null ) {
      tiles[ val ] = 0;
    }
    tiles[ val ]++;
  }

  // delete random elements
  for ( i = 0; i < screen.width() / cellw * screen.height() / cellh; i++ ) {
    if ( Math.random() > 0.4 ) {
      if ( tiles[ grid[i] ] > grid[i] ) {
        tiles[ grid[i] ]--;
        grid[i] = -1;
      }
    }
  }

  // delete necessary elements
  for ( i = 2; i <= maxval; i++ ) {
    if ( tiles[i] > i ) {
      var count = tiles[i];
      for ( j = 0; j < gridw * gridh && tiles[i] > i; j++ ) {
        if ( grid[j] == i ) {
          grid[j] = -1;
          tiles[i]--;
        }
      }
    }
  }

  // add necessary elements
  for ( i = 2; i <= maxval; i++ ) {
    if ( tiles[i] < i ) {
      var count = tiles[i];
      while ( tiles[i] < i ) {
        var randIndex = Math.round( Math.random() * gridw * gridh );
        if ( grid[ randIndex ] == -1 ) {
          grid[ randIndex ] = i;
          tiles[i]++;
        }
      }
    }
  }

  // count active cells
  activeCells = 0;
  for ( i = 2; i <= maxval; i++ ) {
    activeCells += tiles[i];
  }

  grid.get = function( x, y ) {
    return y >= 0 && x >= 0 && x < gridw && y < gridh ? this[y * gridw + x] : -2;
  }

  draw();

  Hammer( document.getElementById( 'screen' ), { preventDefault: true } ).on( "dragend", pollDrag );

  document.onkeydown = pollKey;
}
